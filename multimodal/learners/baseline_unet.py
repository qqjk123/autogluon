import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, Lambdad, DivisiblePadd, ToTensord
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot

# 1. cuDNN 自动调优
torch.backends.cudnn.benchmark = True

class BaselineUNetSegmenter:
    def __init__(self, save_dir: str, lr: float = 1e-4,
                 val_split: float = 0.2, in_channels: int = 4):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.val_split = val_split
        self.in_channels = in_channels

        # label mapping and classes
        self.LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
        self.num_classes = len(self.LABEL_MAP)

        # model, optimizer, loss, metrics
        self.model = None
        self.optimizer = None
        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        self.val_metric = DiceMetric(include_background=False, reduction="mean")
        # test metrics
        self.test_dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )

    def _build_model(self):
        self.model = UNet(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler()

    def _get_dataloader(self, data_list, batch_size, shuffle, num_workers):
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Lambdad(keys="label", func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=3000,
                b_min=0.0, b_max=1.0, clip=True
            ),
            DivisiblePadd(keys=["image", "label"], k=8),
            ToTensord(keys=["image", "label"]),
        ]
        ds = CacheDataset(data=data_list, transform=transforms)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
            collate_fn=pad_list_data_collate
        )

    def fit_from_nnunet(self, nnunet_raw: str, task_id: str,
                        epochs: int = 50, batch_size: int = 1,
                        num_workers: int = 4):
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        examples = info["training"]
        split = int(len(examples) * (1 - self.val_split))
        train_exs, val_exs = examples[:split], examples[split:]
        self._build_model()

        train_data = [
            {"image": [str(ds / "imagesTr" / fn) for fn in ex["image"]],
             "label": str(ds / "labelsTr" / ex["label"])}
            for ex in train_exs
        ]
        val_data = [
            {"image": [str(ds / "imagesTr" / fn) for fn in ex["image"]],
             "label": str(ds / "labelsTr" / ex["label"])}
            for ex in val_exs
        ]
        train_loader = self._get_dataloader(train_data, batch_size, True, num_workers)
        val_loader = self._get_dataloader(val_data, batch_size, False, num_workers)

        best_val = 0.0
        for ep in range(1, epochs + 1):
            # training
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                lbl = batch["label"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                # autocast
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    loss = self.loss_fn(preds, lbl)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            # validation
            self.model.eval()
            self.val_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device, non_blocking=True)
                    lbl = batch["label"].to(self.device, non_blocking=True)
                    preds = sliding_window_inference(
                        imgs, roi_size=(128, 128, 128), sw_batch_size=1,
                        predictor=self.model, overlap=0.5
                    )
                    pred_lbl = torch.argmax(preds, dim=1, keepdim=True)
                    oh_pred = one_hot(pred_lbl, num_classes=self.num_classes)
                    oh_true = one_hot(lbl, num_classes=self.num_classes)
                    self.val_metric(y_pred=oh_pred, y=oh_true)
                val_dice = self.val_metric.aggregate().item()
                self.val_metric.reset()

            print(f"[Epoch {ep}/{epochs}] train loss: {total_loss/len(train_loader):.4f} | val Dice: {val_dice:.4f}")
            if val_dice > best_val:
                best_val = val_dice
                torch.save(self.model.state_dict(), str(self.save_dir / "best_model.pt"))

        print(f"训练完成，最佳 val Dice={best_val:.4f}")

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit_from_nnunet() first.")
        self.model.eval()
        with torch.no_grad():
            return self.model(img.to(self.device))

    def evaluate_from_nnunet(self, nnunet_raw: str, task_id: str,
                             batch_size: int = 1, num_workers: int = 4) -> dict:
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        test_list = info.get("test", info.get("testing", []))

        # test transforms (no random augment)
        test_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Lambdad(keys="label", func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=3000,
                b_min=0.0, b_max=1.0, clip=True
            ),
            DivisiblePadd(keys=["image", "label"], k=8),
            ToTensord(keys=["image", "label"]),
        ]

        data_list = [
            {"image": [str(ds / "imagesTs" / fn) for fn in ex["image"]],
             "label": str(ds / "labelsTs" / ex["label"])}
            for ex in test_list
        ]
        loader = DataLoader(
            CacheDataset(data=data_list, transform=test_transforms, cache_rate=0.0),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=pad_list_data_collate
        )

        per_case_dice = []
        per_case_hd95 = []
        tp = tn = fp = fn = 0

        self.test_hd95_metric.reset()
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(self.device)
                lbl = batch["label"].to(self.device)
                preds = sliding_window_inference(
                    imgs, roi_size=(128, 128, 128), sw_batch_size=1,
                    predictor=self.model, overlap=0.5
                )
                pred_lbl = torch.argmax(preds, dim=1)

                # confusion
                y_pred_bin = (pred_lbl > 0)
                y_true_bin = (lbl > 0)
                tp += int((y_pred_bin & y_true_bin).sum())
                tn += int((~y_pred_bin & ~y_true_bin).sum())
                fp += int((y_pred_bin & ~y_true_bin).sum())
                fn += int((~y_pred_bin & y_true_bin).sum())

                # one-hot
                pred_oh = one_hot(pred_lbl.unsqueeze(1), num_classes=self.num_classes)
                true_oh = one_hot(lbl, num_classes=self.num_classes)

                # Dice
                self.test_dice_metric.reset()
                self.test_dice_metric(y_pred=pred_oh, y=true_oh)
                per_case_dice.append(self.test_dice_metric.aggregate().item())

                # HD95
                self.test_hd95_metric.reset()
                self.test_hd95_metric(y_pred=pred_oh, y=true_oh)
                per_case_hd95.append(self.test_hd95_metric.aggregate().item())

        mean_dice = float(np.mean(per_case_dice))
        mean_hd95 = float(np.mean(per_case_hd95))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        def _bootstrap(values, n_bootstraps=1000, seed=42):
            rng = np.random.default_rng(seed)
            n = len(values)
            means = [rng.choice(values, size=n, replace=True).mean() for _ in range(n_bootstraps)]
            var = float(np.var(means, ddof=1))
            ci_low, ci_high = np.percentile(means, [2.5, 97.5])
            return var, ci_low, ci_high

        dice_var, dice_ci_l, dice_ci_u = _bootstrap(per_case_dice)
        hd95_var, hd95_ci_l, hd95_ci_u = _bootstrap(per_case_hd95)
        sens_var, sens_ci_l, sens_ci_u = _bootstrap([sensitivity])
        spec_var, spec_ci_l, spec_ci_u = _bootstrap([specificity])

        print(f"Test Results → Dice: {mean_dice:.4f} ± {dice_var:.6f} (95% CI [{dice_ci_l:.3f}, {dice_ci_u:.3f}])")
        print(f"             HD95: {mean_hd95:.2f} ± {hd95_var:.4f} (95% CI [{hd95_ci_l:.2f}, {hd95_ci_u:.2f}])")
        print(f"       Sensitivity: {sensitivity:.4f} ± {sens_var:.6f} (95% CI [{sens_ci_l:.3f}, {sens_ci_u:.3f}])")
        print(f"       Specificity: {specificity:.4f} ± {spec_var:.6f} (95% CI [{spec_ci_l:.3f}, {spec_ci_u:.3f}])")

        return {
            "dice": mean_dice,
            "dice_var": dice_var,
            "dice_ci_lower": dice_ci_l,
            "dice_ci_upper": dice_ci_u,
            "hd95": mean_hd95,
            "hd95_var": hd95_var,
            "hd95_ci_lower": hd95_ci_l,
            "hd95_ci_upper": hd95_ci_u,
            "sensitivity": sensitivity,
            "sens_var": sens_var,
            "sens_ci_lower": sens_ci_l,
            "sens_ci_upper": sens_ci_u,
            "specificity": specificity,
            "spec_var": spec_var,
            "spec_ci_lower": spec_ci_l,
            "spec_ci_upper": spec_ci_u,
        }
