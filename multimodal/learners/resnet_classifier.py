import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.networks.nets import resnet18

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        print(f"Initializing ResNetClassifier(in_channels={in_channels}, num_classes={num_classes})")
        self.backbone = resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
            pretrained=False
        )
    def forward(self, x):
        return self.backbone(x)

class ResNetPredictor:
    def __init__(
        self,
        metadata_csv: str,
        output_dir: str,
        in_channels: int = 4,
        batch_size: int = 16,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 50,
        patience: int = 10,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        val_split: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
        n_bootstrap: int = 1000,
        ci_alpha: float = 0.05
    ):
        # ——————— Initialization ———————
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}, batch_size: {batch_size}, AMP: {use_amp}")

        # params
        self.meta_csv = metadata_csv
        self.out_dir   = output_dir
        self.C         = in_channels
        self.bs        = batch_size
        self.lr        = lr
        self.wd        = weight_decay
        self.epochs    = max_epochs
        self.patience  = patience
        self.acc_steps = accumulation_steps
        self.amp       = use_amp
        self.val_split = val_split
        self.nw        = num_workers
        self.pm        = pin_memory
        self.n_boot    = n_bootstrap
        self.alpha     = ci_alpha

        # read metadata & train/val/test splits
        df = pd.read_csv(self.meta_csv, dtype={"case_id": str}, encoding="utf-8-sig")
        ids_tr = df[df.split=="train"].case_id.unique().tolist()
        ids_ts = df[df.split=="test"].case_id.unique().tolist()
        labels = df.drop_duplicates("case_id").set_index("case_id")["label"].to_dict()
        if self.val_split>0 and len(ids_tr)>1:
            tr, vl = train_test_split(
                ids_tr,
                test_size=self.val_split,
                stratify=[labels[i] for i in ids_tr],
                random_state=42
            )
        else:
            tr, vl = ids_tr, []
        self.train_ids, self.val_ids, self.test_ids = tr, vl, ids_ts
        print(f"Found {len(tr)} train, {len(vl)} val, {len(ids_ts)} test cases.")

        # compute a fixed crop‐shape across all cases
        self._compute_global_shape(tr + vl + ids_ts)

    def _compute_global_shape(self, case_ids):
        df = pd.read_csv(self.meta_csv, dtype={"case_id": str}, encoding="utf-8-sig")
        mins = []
        for cid in case_ids:
            g = df[df.case_id==cid]
            shapes = []
            for m in ["FLAIR","T1w","T1wCE","T2w"]:
                fp = g[g.modality==m].image_path.iloc[0]
                shapes.append(nib.load(fp).shape)
            shapes = np.stack(shapes,0)
            mins.append(shapes.min(axis=0))
        self.global_shape = tuple(int(x) for x in np.stack(mins,0).min(axis=0))
        print(f"Global crop shape set to D,H,W = {self.global_shape}")

    def _load_cases(self, case_ids):
        df = pd.read_csv(self.meta_csv, dtype={"case_id": str}, encoding="utf-8-sig")
        Dg, Hg, Wg = self.global_shape
        X, y = [], []
        for cid in case_ids:
            g = df[df.case_id==cid]
            vols = []
            for m in ["FLAIR","T1w","T1wCE","T2w"]:
                v = nib.load(g[g.modality==m].image_path.iloc[0]).get_fdata().astype(np.float32)
                d,h,w = v.shape
                d0 = (d - Dg)//2; h0 = (h - Hg)//2; w0 = (w - Wg)//2
                vols.append(v[d0:d0+Dg, h0:h0+Hg, w0:w0+Wg])
            X.append(np.stack(vols,0)); y.append(int(g.label.iloc[0]))
        X = np.stack(X,0); y = np.array(y, dtype=np.int64)
        return X, y

    def _make_loader(self, X, y, shuffle):
        X = (X - X.min())/(X.max()-X.min()+1e-8)
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds,
                          batch_size=self.bs,
                          shuffle=shuffle,
                          num_workers=self.nw,
                          pin_memory=self.pm)

    def fit(self):
        # ——————— Training Loop ———————
        X_tr, y_tr = self._load_cases(self.train_ids)
        tr_loader = self._make_loader(X_tr, y_tr, True)
        if self.val_ids:
            X_vl, y_vl = self._load_cases(self.val_ids)
            vl_loader = self._make_loader(X_vl, y_vl, False)

        nc = len(np.unique(y_tr))
        self.model = ResNetClassifier(self.C, nc).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                      patience=5, min_lr=1e-6)
        scaler = GradScaler()
        
        best_auc = -np.inf
        wait = 0

        for epoch in range(1, self.epochs+1):
            print(f"\n=== Epoch {epoch}/{self.epochs} ===")
            # — training —
            self.model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            for i, (Xb, yb) in enumerate(tr_loader,1):
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                with autocast(self.amp):
                    logits = self.model(Xb)
                    loss = nn.CrossEntropyLoss()(logits, yb)/self.acc_steps
                scaler.scale(loss).backward()
                if i % self.acc_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                running_loss += loss.item()*self.acc_steps
            print(f"  Train loss: {running_loss/len(tr_loader):.4f}")

            # — validation —
            if self.val_ids:
                print("  Validating…")
                self.model.eval()
                all_preds, all_trues, all_probs = [], [], []
                with torch.no_grad():
                    for Xb,yb in vl_loader:
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        out = torch.softmax(self.model(Xb),1)
                        preds = out.argmax(1)
                        all_preds.append(preds.cpu().numpy())
                        all_trues.append(yb.cpu().numpy())
                        all_probs.append(out.cpu().numpy()[:,1] if nc==2 else out.cpu().numpy())
                preds = np.concatenate(all_preds)
                trues = np.concatenate(all_trues)
                probs = np.concatenate(all_probs)
                acc = accuracy_score(trues, preds)
                auc = (roc_auc_score(trues, probs)
                       if nc==2 else roc_auc_score(trues, probs, multi_class="ovr", average="macro"))
                print(f"  Val Acc={acc:.4f}, AUC={auc:.4f}")
                scheduler.step(auc)

                # save best‐AUC model
                if auc > best_auc:
                    best_auc = auc; wait = 0
                    best_path = os.path.join(self.out_dir, f"best_auc_epoch{epoch}.pt")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  → New best AUC; model saved to {best_path}")
                    self.best_model_path = best_path
                else:
                    wait += 1
                    if wait >= self.patience:
                        print("  Early stopping on AUC")
                        break

        # always save final too
        final_path = os.path.join(self.out_dir, "final.pt")
        torch.save(self.model.state_dict(), final_path)
        print(f"Training done. Final model at {final_path}")

    def _bootstrap_eval(self, trues, preds, probs):
        """Compute bootstrap CIs for acc, auc, f1, recall."""
        rng = np.random.RandomState(42)
        metrics = {"acc":[], "auc":[], "f1":[],"sens":[]}
        n = len(trues)
        for _ in range(self.n_boot):
            idx = rng.randint(0, n, size=n)
            yt = trues[idx]
            yp = preds[idx]
            pv = probs[idx] if probs.ndim==1 else probs[idx,1]
            metrics["acc"].append(accuracy_score(yt, yp))
            metrics["auc"].append(roc_auc_score(yt, pv))
            metrics["f1"].append(f1_score(yt, yp, average="macro"))
            metrics["sens"].append(recall_score(yt, yp, average="macro"))
        out = {}
        for k, vals in metrics.items():
            m = float(np.mean(vals))
            lo = float(np.percentile(vals,    100*self.alpha/2))
            hi = float(np.percentile(vals,100*(1-self.alpha/2)))
            out[f"{k}"]    = m
            out[f"{k}_ci"] = (lo, hi)
        return out

    def evaluate(self):
        """Loads best‐AUC checkpoint, runs a single pass plus bootstrap."""
        print("\n=== Bootstrap test evaluation ===")
        # load the best‐AUC model
        ckpt = getattr(self, "best_model_path", None)
        if ckpt is None:
            raise RuntimeError("No best‐AUC model found. Run fit() first.")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        X_ts, y_ts = self._load_cases(self.test_ids)
        loader = self._make_loader(X_ts, y_ts, False)

        all_preds, all_trues, all_probs = [], [], []
        with torch.no_grad():
            for Xb,yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                out = torch.softmax(self.model(Xb),1)
                preds = out.argmax(1)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
                # for binary: take class-1 prob; for multi: keep full array
                all_probs.append(out.cpu().numpy()[:,1] 
                                 if out.shape[1]==2 else out.cpu().numpy())
        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        probs = np.concatenate(all_probs)

        results = self._bootstrap_eval(trues, preds, probs)
        print("Test set (bootstrap):")
        print(f"  Acc       = {results['acc']:.4f} ± [{results['acc_ci'][0]:.4f}, {results['acc_ci'][1]:.4f}]")
        print(f"  AUC       = {results['auc']:.4f} ± [{results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f}]")
        print(f"  F1-score  = {results['f1']:.4f} ± [{results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f}]")
        print(f"  Sensitivity = {results['sens']:.4f} ± [{results['sens_ci'][0]:.4f}, {results['sens_ci'][1]:.4f}]")
        return results

# ——— USAGE ———
if __name__=="__main__":
    predictor = ResNetPredictor(
        metadata_csv = "metadata.csv",
        output_dir   = "output/resnet_bootstrap",
        max_epochs   = 50,
        patience     = 10,
        n_bootstrap  = 1000,
        ci_alpha     = 0.05
    )
    predictor.fit()
    test_metrics = predictor.evaluate()
