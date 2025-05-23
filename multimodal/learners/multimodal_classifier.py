# classification_learner.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    DivisiblePadd, ResizeWithPadOrCropd, ScaleIntensityRanged,
    ConcatItemsd, ToTensord
)


class MultiModalClassifier(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        num_classes: int,
        encoder_feature_size: int = 32,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, encoder_feature_size, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(encoder_feature_size, encoder_feature_size, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
            )
            for _ in range(in_channels)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * encoder_feature_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i, enc in enumerate(self.encoders):
            xi = x[:, i:i+1].contiguous()
            hi = enc(xi)
            feats.append(hi.view(x.shape[0], -1))
        h = torch.cat(feats, dim=1)
        return self.classifier(h)


class MultiModalClassificationLearner:
    def __init__(
        self,
        save_dir: str,
        metadata_csv: str,
        modalities: list[str] = ["FLAIR", "T1w", "T1wCE", "T2w"],
        img_size: tuple[int, int, int] = (128, 128, 128),
        lr: float = 1e-4,
        batch_size: int = 4,
        val_split: float = 0.2,
        epochs: int = 50,
        num_workers: int = 4,
        n_bootstrap: int = 1000,
        ci_alpha: float = 0.05,
        device: str | torch.device = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_csv = metadata_csv
        self.modalities = modalities
        self.img_size = img_size
        self.lr = lr
        self.batch_size = batch_size
        self.val_split = val_split
        self.epochs = epochs
        self.num_workers = num_workers
        self.n_bootstrap = n_bootstrap
        self.ci_alpha = ci_alpha
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model: nn.Module
        self.optimizer: optim.Optimizer
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_data_list(self, df: pd.DataFrame) -> list[dict]:
        data_list = []
        grouped = df.groupby('case_id')
        for case_id, group in grouped:
            item = {}
            for mod in self.modalities:
                rows = group[group['modality'] == mod]
                if rows.empty:
                    raise KeyError(f"Modality {mod} not found for case {case_id}")
                item[mod] = rows['image_path'].values[0]
            item['label'] = int(group['label'].values[0])
            item['split'] = group['split'].values[0]
            data_list.append(item)
        return data_list

    def _get_dataloaders(self):
        df = pd.read_csv(self.metadata_csv, dtype={'case_id': str})
        data_list = self._prepare_data_list(df)
        train_list = [x for x in data_list if x['split'] == 'train']
        val_list   = [x for x in data_list if x['split'] == 'val']
        test_list  = [x for x in data_list if x['split'] == 'test']

        if not (train_list and val_list and test_list):
            all_list = data_list
            train_list, test_list = train_test_split(
                all_list, test_size=self.val_split,
                stratify=[x['label'] for x in all_list], random_state=42
            )
            train_list, val_list = train_test_split(
                train_list, test_size=self.val_split,
                stratify=[x['label'] for x in train_list], random_state=42
            )

        def make_loader(items, shuffle: bool):
            transforms = [
                LoadImaged(keys=self.modalities),
                EnsureChannelFirstd(keys=self.modalities),
                Spacingd(keys=self.modalities, pixdim=(1.0,1.0,1.0), mode='bilinear'),
                Orientationd(keys=self.modalities, axcodes='RAS'),
                DivisiblePadd(keys=self.modalities, k=16),
                ResizeWithPadOrCropd(keys=self.modalities, spatial_size=self.img_size),
                ScaleIntensityRanged(keys=self.modalities, a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
                ConcatItemsd(keys=self.modalities, name='image', dim=0),
                ToTensord(keys=['image','label']),
            ]
            ds = CacheDataset(data=items, transform=transforms)
            return DataLoader(
                ds, batch_size=self.batch_size, shuffle=shuffle,
                num_workers=self.num_workers, pin_memory=True,
                persistent_workers=True, prefetch_factor=2,
                collate_fn=pad_list_data_collate
            )

        return make_loader(train_list, True), make_loader(val_list, False), make_loader(test_list, False)

    def fit(self):
        train_loader, val_loader, test_loader = self._get_dataloaders()
        print(f"Dataset sizes -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        num_classes = len({x['label'] for x in self._prepare_data_list(pd.read_csv(self.metadata_csv))})
        self.model = MultiModalClassifier(
            in_channels=len(self.modalities), num_classes=num_classes
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch in train_loader:
                imgs = batch['image'].to(self.device, non_blocking=True)
                lbls = batch['label'].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss = self.criterion(logits, lbls)
                loss.backward()
                self.optimizer.step()
            self.model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch['image'].to(self.device, non_blocking=True)
                    lbls = batch['label'].to(self.device, non_blocking=True)
                    logits = self.model(imgs)
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                    trues.append(lbls.cpu().numpy())
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            acc = accuracy_score(trues, preds)
            print(f"Epoch {epoch}/{self.epochs} - Val Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pt")
        print(f"Training complete, best Val Accuracy = {best_acc:.4f}")

    def evaluate(self):
        _, _, test_loader = self._get_dataloaders()
        self.model.load_state_dict(torch.load(self.save_dir / "best_model.pt", map_location=self.device))
        self.model.eval()
        all_preds, all_trues, all_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch['image'].to(self.device, non_blocking=True)
                lbls = batch['label'].to(self.device, non_blocking=True)
                logits = self.model(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                if probs.shape[1] == 2:
                    all_probs.append(probs[:,1])
                else:
                    all_probs.append(probs)
                all_preds.append(np.argmax(probs, axis=1))
                all_trues.append(lbls.cpu().numpy())
        trues = np.concatenate(all_trues)
        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)
        results = self._bootstrap_eval(trues, preds, probs)
        print("Test set (bootstrap):")
        print(f"  Acc         = {results['acc']:.4f} ± [{results['acc_ci'][0]:.4f}, {results['acc_ci'][1]:.4f}] (std ~{(results['acc_ci'][1]-results['acc_ci'][0])/2:.4f})")
        print(f"  AUC         = {results['auc']:.4f} ± [{results['auc_ci'][0]:.4f}, {results['auc_ci'][1]:.4f}] (std ~{(results['auc_ci'][1]-results['auc_ci'][0])/2:.4f})")
        print(f"  F1-score    = {results['f1']:.4f} ± [{results['f1_ci'][0]:.4f}, {results['f1_ci'][1]:.4f}] (std ~{(results['f1_ci'][1]-results['f1_ci'][0])/2:.4f})")
        print(f"  Sensitivity = {results['sens']:.4f} ± [{results['sens_ci'][0]:.4f}, {results['sens_ci'][1]:.4f}] (std ~{(results['sens_ci'][1]-results['sens_ci'][0])/2:.4f})")
        return results

    def _bootstrap_eval(self, trues, preds, probs):
        n = len(trues)
        idx = np.arange(n)
        stats = {
            'acc': accuracy_score(trues, preds),
            'auc': roc_auc_score(trues, probs) if probs.ndim==1 else roc_auc_score(trues, probs, multi_class='ovr'),
            'f1': f1_score(trues, preds, average='binary' if preds.ndim==1 else 'macro'),
            'sens': recall_score(trues, preds, average='binary' if preds.ndim==1 else 'macro')
        }
        ci = {k + '_ci': [] for k in stats}
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(idx, size=n, replace=True)
            t_s, p_s, prob_s = trues[sample], preds[sample], probs[sample]
            ci['acc_ci'].append(accuracy_score(t_s, p_s))
            ci['auc_ci'].append(roc_auc_score(t_s, prob_s) if prob_s.ndim==1 else roc_auc_score(t_s, prob_s, multi_class='ovr'))
            ci['f1_ci'].append(f1_score(t_s, p_s, average='binary' if p_s.ndim==1 else 'macro'))
            ci['sens_ci'].append(recall_score(t_s, p_s, average='binary' if p_s.ndim==1 else 'macro'))
        results = {}
        for k, v in stats.items():
            lower = np.percentile(ci[k + '_ci'], 100 * self.ci_alpha / 2)
            upper = np.percentile(ci[k + '_ci'], 100 * (1 - self.ci_alpha / 2))
            results[k] = v
            results[k + '_ci'] = (lower, upper)
        return results
