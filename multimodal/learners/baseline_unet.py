# baseline_unet_segmenter.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    Lambdad,
    DivisiblePadd,
    ToTensord
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot


class BaselineUNetSegmenter:
    def __init__(
        self,
        save_dir: str,
        lr: float = 1e-4,
        val_split: float = 0.2,
        in_channels: int = 4
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.val_split = val_split
        self.in_channels = in_channels

        # 现在我们明确有 4 个类别：0,1,2,3
        self.LABEL_MAP = {0:0, 1:1, 2:2, 3:3}
        self.num_classes = len(self.LABEL_MAP)

        self.model = None
        self.optimizer = None
        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")

    def _build_model(self):
        self.model = UNet(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_dataloader(self, data_list, batch_size, shuffle, num_workers):
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"],
                     pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # 保证所有标签都在 0–3 范围内
            Lambdad(keys="label", func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
            ScaleIntensityRanged(keys=["image"],
                                 a_min=0, a_max=3000,
                                 b_min=0.0, b_max=1.0,
                                 clip=True),
            DivisiblePadd(keys=["image", "label"], k=8),
            ToTensord(keys=["image", "label"]),
        ]

        ds = CacheDataset(data=data_list, transform=transforms)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate,
        )

    def fit_from_nnunet(
        self,
        nnunet_raw: str,
        task_id: str,
        epochs: int = 50,
        batch_size: int = 1,
        num_workers: int = 4
    ):
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        examples = info["training"]

        split = int(len(examples) * (1 - self.val_split))
        train_exs, val_exs = examples[:split], examples[split:]

        # 构建模型（现在使用固定的 4 类）
        self._build_model()

        train_data = [
            {"image": [str(ds/"imagesTr"/fn) for fn in ex["image"]],
             "label": str(ds/"labelsTr"/ex["label"])}
            for ex in train_exs
        ]
        val_data = [
            {"image": [str(ds/"imagesTr"/fn) for fn in ex["image"]],
             "label": str(ds/"labelsTr"/ex["label"])}
            for ex in val_exs
        ]

        train_loader = self._get_dataloader(train_data, batch_size, True, num_workers)
        val_loader   = self._get_dataloader(val_data,   batch_size, False, num_workers)

        best_val = 0.0
        for ep in range(1, epochs+1):
            # 训练
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device)
                lbl  = batch["label"].to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(imgs)
                loss  = self.loss_fn(preds, lbl)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # 验证
            self.model.eval()
            self.metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device)
                    lbl  = batch["label"].to(self.device)
                    preds = sliding_window_inference(
                        imgs, roi_size=(128,128,128), sw_batch_size=1,
                        predictor=self.model, overlap=0.5
                    )
                    pred_lbl = torch.argmax(preds, dim=1, keepdim=True)
                    oh_pred  = one_hot(pred_lbl, num_classes=self.num_classes)
                    oh_true  = one_hot(lbl.unsqueeze(1), num_classes=self.num_classes)
                    self.metric(y_pred=oh_pred, y=oh_true)
                val_dice = self.metric.aggregate().item()
                self.metric.reset()

            print(f"Epoch {ep}/{epochs} - train loss: {total_loss/len(train_loader):.4f}, val Dice: {val_dice:.4f}")
            if val_dice > best_val:
                best_val = val_dice
                torch.save(self.model.state_dict(), str(self.save_dir/"best_model.pt"))

        print(f"Training complete. Best val Dice: {best_val:.4f}")

    def predict(
        self,
        img_tensor: torch.Tensor,
        roi_size=(128, 128, 128),
        overlap: float = 0.5
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit_from_nnunet first.")
        self.model.eval()
        with torch.no_grad():
            return sliding_window_inference(
                img_tensor.to(self.device),
                roi_size, sw_batch_size=1,
                predictor=self.model, overlap=overlap,
            )

    def evaluate_from_nnunet(
        self,
        nnunet_raw: str,
        task_id: str,
        batch_size: int = 1,
        num_workers: int = 4
    ) -> dict:
        ds   = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds/"dataset.json"))
        test_exs = info.get("test", info.get("testing", []))

        loader = DataLoader(
            CacheDataset(data=[
                {"image":[str(ds/"imagesTs"/fn) for fn in ex["image"]],
                 "label": str(ds/"labelsTs"/ex["label"])}
                for ex in test_exs
            ], transform=[
                LoadImaged(keys=["image","label"]),
                EnsureChannelFirstd(keys=["image","label"]),
                Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
                Orientationd(keys=["image","label"], axcodes="RAS"),
                Lambdad(keys="label", func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
                ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
                DivisiblePadd(keys=["image","label"], k=8),
                ToTensord(keys=["image","label"]),
            ]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate,
        )

        results = {"dice": [], "hd95": [], "sens": [], "spec": []}
        tp = tn = fp = fn = 0
        dice_m = DiceMetric(include_background=False, reduction="mean")
        hd95_m = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        with torch.no_grad():
            self.model.eval()
            for batch in loader:
                imgs = batch["image"].to(self.device)
                lbl  = batch["label"].to(self.device)
                preds = sliding_window_inference(imgs,(128,128,128),1,self.model,overlap=0.5)
                pred_lbl = torch.argmax(preds, dim=1)

                # 二分类灵敏度/特异度统计
                y_pred_b = pred_lbl > 0
                y_true_b = lbl > 0
                tp += int((y_pred_b & y_true_b).sum())
                tn += int(((~y_pred_b) & (~y_true_b)).sum())
                fp += int((y_pred_b & (~y_true_b)).sum())
                fn += int(((~y_pred_b) & y_true_b).sum())

                one_pred = one_hot(pred_lbl.unsqueeze(1), num_classes=self.num_classes)
                one_true = one_hot(lbl, num_classes=self.num_classes)

                dice_m(y_pred=one_pred, y=one_true)
                results["dice"].append(dice_m.aggregate().item())
                dice_m.reset()

                hd95_m(y_pred=one_pred, y=one_true)
                results["hd95"].append(hd95_m.aggregate().item())
                hd95_m.reset()

                sens = tp / (tp + fn) if tp + fn > 0 else 0.0
                spec = tn / (tn + fp) if tn + fp > 0 else 0.0
                results["sens"].append(sens)
                results["spec"].append(spec)

        # 计算平均、方差和 95% CI
        def _bootstrap(vals, n=1000, seed=0):
            rng = np.random.default_rng(seed)
            m = len(vals)
            means = [rng.choice(vals, m, replace=True).mean() for _ in range(n)]
            return (float(np.mean(means)),
                    float(np.var(means, ddof=1)),
                    tuple(np.percentile(means, [2.5, 97.5])))

        summary = {}
        for k in ["dice","hd95","sens","spec"]:
            mean, var, (lo, hi) = _bootstrap(results[k])
            print(f"{k.upper()}: {mean:.4f} var={var:.6f} CI=[{lo:.4f},{hi:.4f}]")
            summary[f"{k}"]     = mean
            summary[f"{k}_var"] = var
            summary[f"{k}_ci"]  = (lo, hi)

        return summary
