#!/usr/bin/env python3
"""
Parallelized training script for 3D ResNet classification using PyTorch DataLoader.
Method 1: loads and preprocesses each case in Dataset workers.
"""
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from monai.networks.nets import resnet18
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CaseDataset(Dataset):
    def __init__(self, metadata_csv, case_ids, global_shape, normalize=True):
        self.df = pd.read_csv(metadata_csv, dtype={"case_id": str}, encoding="utf-8-sig")
        self.ids = case_ids
        self.Dg, self.Hg, self.Wg = global_shape
        self.normalize = normalize

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        g = self.df[self.df.case_id == cid]
        vols = []
        for m in ["FLAIR", "T1w", "T1wCE", "T2w"]:
            arr = nib.load(g[g.modality == m].image_path.iloc[0]).get_fdata().astype(np.float32)
            d, h, w = arr.shape
            d0 = max((d - self.Dg) // 2, 0)
            h0 = max((h - self.Hg) // 2, 0)
            w0 = max((w - self.Wg) // 2, 0)
            crop = arr[d0:d0+self.Dg, h0:h0+self.Hg, w0:w0+self.Wg]
            pad_d = max(self.Dg - crop.shape[0], 0)
            pad_h = max(self.Hg - crop.shape[1], 0)
            pad_w = max(self.Wg - crop.shape[2], 0)
            padded = np.pad(
                crop,
                ((pad_d//2, pad_d - pad_d//2),
                 (pad_h//2, pad_h - pad_h//2),
                 (pad_w//2, pad_w - pad_w//2)),
                mode="constant",
                constant_values=0
            )
            vols.append(padded)
        X = np.stack(vols, axis=0)
        if self.normalize:
            X = (X - X.min()) / (X.max() - X.min() + 1e-8)
        y = int(g.label.iloc[0])
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
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
        metadata_csv,
        output_dir,
        in_channels=4,
        batch_size=16,
        lr=2e-4,
        weight_decay=1e-5,
        max_epochs=50,
        patience=10,
        accumulation_steps=1,
        use_amp=True,
        val_split=0.2,
        num_workers=4,
        pin_memory=True,
        n_bootstrap=1000,
        ci_alpha=0.05
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}, batch_size: {batch_size}, AMP: {use_amp}, workers: {num_workers}")

        # 参数
        self.meta_csv = metadata_csv
        self.out_dir = output_dir
        self.C = in_channels
        self.bs = batch_size
        self.lr = lr
        self.wd = weight_decay
        self.epochs = max_epochs
        self.patience = patience
        self.acc_steps = accumulation_steps
        self.amp = use_amp
        self.val_split = val_split
        self.nw = num_workers
        self.pm = pin_memory
        self.n_boot = n_bootstrap
        self.alpha = ci_alpha
        self.best_model_path = None  # Track path to best checkpoint

        # 读取 splits: test set 固定为 metadata 中标记为 'test' 的 case，其余按 val_split 划分为 train/val
        df_all = pd.read_csv(self.meta_csv, dtype={"case_id": str}, encoding="utf-8-sig")
        if 'split' in df_all.columns:
            df_test = df_all[df_all['split'] == 'test']
            df_tv = df_all[df_all['split'] != 'test']
        else:
            # 若无 split 列，则将所有数据用于 train/val，test 使用全部
            df_test = df_all.copy()
            df_tv = df_all.copy()
        self.test_ids = df_test.case_id.unique().tolist()
        ids_tv = df_tv.case_id.unique().tolist()
        labels_tv = df_tv.drop_duplicates('case_id').set_index('case_id')['label'].to_dict()
        self.train_ids, self.val_ids = train_test_split(
            ids_tv,
            test_size=self.val_split,
            stratify=[labels_tv[i] for i in ids_tv],
            random_state=42
        )
        print(f"Found {len(self.train_ids)} train, {len(self.val_ids)} val, {len(self.test_ids)} test cases.")

        # 计算 global_shape
        self.global_shape = self._compute_global_shape()

    def _compute_global_shape(self):
        df = pd.read_csv(self.meta_csv, dtype={"case_id": str}, encoding="utf-8-sig")
        shapes = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Inspect shapes", unit="img"):
            try:
                shapes.append(nib.load(row.image_path).shape)
            except Exception:
                pass
        arr = np.array(shapes)
        means = np.round(arr.mean(axis=0)).astype(int)
        div = 16
        target = ((means + div - 1) // div) * div
        print(f"Global shape = {tuple(target)}")
        return tuple(target)

    def fit(self):
        # 构建 DataLoader
        train_ds = CaseDataset(self.meta_csv, self.train_ids, self.global_shape)
        tr_loader = DataLoader(
            train_ds, batch_size=self.bs, shuffle=True,
            num_workers=self.nw, pin_memory=self.pm
        )
        val_ds = CaseDataset(self.meta_csv, self.val_ids, self.global_shape)
        vl_loader = DataLoader(
            val_ds, batch_size=self.bs, shuffle=False,
            num_workers=self.nw, pin_memory=self.pm
        )

        # 计算类别数
        df_meta = pd.read_csv(self.meta_csv, dtype={"case_id": str}, encoding="utf-8-sig")
        labels = df_meta.drop_duplicates("case_id").set_index("case_id")["label"].to_dict()
        nc = len(set(labels.values()))
        self.model = ResNetClassifier(self.C, nc).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
        scaler = GradScaler()

        best_acc = -np.inf
        wait = 0
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            self.model.train()
            train_loss = 0
            for i, (Xb, yb) in enumerate(tqdm(tr_loader, desc="Train", unit="batch"), 1):
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                with autocast(self.amp):
                    logits = self.model(Xb)
                    loss = nn.CrossEntropyLoss()(logits, yb) / self.acc_steps
                scaler.scale(loss).backward()
                if i % self.acc_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                train_loss += loss.item() * self.acc_steps
            print(f"Train loss: {train_loss/len(tr_loader):.4f}")

            # 验证
            self.model.eval()
            all_preds, all_trues, all_probs = [], [], []
            with torch.no_grad():
                for Xb, yb in tqdm(vl_loader, desc="Val", unit="batch"):
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    out = torch.softmax(self.model(Xb), dim=1)
                    all_preds.append(out.argmax(1).cpu().numpy())
                    all_trues.append(yb.cpu().numpy())
                    if out.shape[1] == 2:
                        all_probs.append(out.cpu().numpy()[:, 1])
                    else:
                        all_probs.append(out.cpu().numpy())
            preds = np.concatenate(all_preds)
            trues = np.concatenate(all_trues)
            probs = np.concatenate(all_probs)
            acc = accuracy_score(trues, preds)
            auc = roc_auc_score(trues, probs) if probs.ndim == 1 else roc_auc_score(trues, probs, multi_class="ovr")
            print(f"Val Acc={acc:.4f}, AUC={auc:.4f}")
            scheduler.step(acc)

            if acc > best_acc:
                best_acc = acc
                wait = 0
                ckpt_path = os.path.join(self.out_dir, "best.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                self.best_model_path = ckpt_path
                print(f"Saved new best to {ckpt_path}.")
            else:
                wait += 1
                if wait >= self.patience:
                    print("Early stop")
                    break

    def evaluate(self):
        """Loads best‐ACC checkpoint, runs DataLoader‐based test pass + bootstrap."""
        print("\n=== Bootstrap test evaluation ===")
        ckpt = getattr(self, "best_model_path", None)
        if ckpt is None:
            raise RuntimeError("No best‐AUC model found. Run fit() first.")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        test_ds = CaseDataset(self.meta_csv, self.test_ids, self.global_shape, normalize=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.nw,
            pin_memory=self.pm,
            persistent_workers=True,
            prefetch_factor=2
        )

        all_preds, all_trues, all_probs = [], [], []
        with torch.no_grad():
            for Xb, yb in tqdm(test_loader, desc="Test", unit="batch"):
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                out = torch.softmax(self.model(Xb), dim=1)
                all_preds.append(out.argmax(1).cpu().numpy())
                all_trues.append(yb.cpu().numpy())
                if out.shape[1] == 2:
                    all_probs.append(out.cpu().numpy()[:, 1])
                else:
                    all_probs.append(out.cpu().numpy())

        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        probs = np.concatenate(all_probs)

        results = self._bootstrap_eval(trues, preds, probs)
        print("Test set (bootstrap):")
        print(f"  Acc         = {results['acc']:.4f} ± [{results['acc_ci'][0]:.4f}, {results['acc_ci'][1]:.4f}]")
        print(f"  AUC         = {results['auc']:.4f} ± [{results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f}]")
        print(f"  F1-score    = {results['f1']:.4f} ± [{results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f}]")
        print(f"  Sensitivity = {results['sens']:.4f} ± [{results['sens_ci'][0]:.4f}, {results['sens_ci'][1]:.4f}]")
        return results

    def _bootstrap_eval(self, trues, preds, probs):
        # implement bootstrap confidence intervals here
        import numpy as np
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
        n = len(trues)
        idx = np.arange(n)
        stats = {'acc': accuracy_score(trues, preds),
                 'auc': roc_auc_score(trues, probs) if probs.ndim == 1 else roc_auc_score(trues, probs, multi_class='ovr'),
                 'f1': f1_score(trues, preds, average='binary' if preds.ndim==1 else 'macro'),
                 'sens': recall_score(trues, preds, average='binary' if preds.ndim==1 else 'macro')}
        ci = {k + '_ci': [] for k in stats}
        for _ in range(self.n_boot):
            sample = np.random.choice(idx, size=n, replace=True)
            t_s, p_s, prob_s = trues[sample], preds[sample], probs[sample]
            ci['acc_ci'].append(accuracy_score(t_s, p_s))
            ci['auc_ci'].append(roc_auc_score(t_s, prob_s) if prob_s.ndim == 1 else roc_auc_score(t_s, prob_s, multi_class='ovr'))
            ci['f1_ci'].append(f1_score(t_s, p_s, average='binary' if p_s.ndim==1 else 'macro'))
            ci['sens_ci'].append(recall_score(t_s, p_s, average='binary' if p_s.ndim==1 else 'macro'))
        results = {}
        for k in stats:
            lower = np.percentile(ci[k+'_ci'], 100 * self.alpha/2)
            upper = np.percentile(ci[k+'_ci'], 100 * (1-self.alpha/2))
            results[k] = stats[k]
            results[k+'_ci'] = (lower, upper)
        return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", required=True, help="Path to metadata CSV")
    p.add_argument("--output_dir", required=True, help="Directory to save models and logs")
    p.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    p.add_argument("--max_epochs", type=int, default=50, help="Maximum training epochs")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return p.parse_args()


