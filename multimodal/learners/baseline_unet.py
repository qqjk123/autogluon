import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

from monai.data import CacheDataset, DataLoader, pad_list_data_collate, SmartCacheDataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    Lambdad,
    ScaleIntensityRanged,
    DivisiblePadd,
    ToTensord,
    # —— 以下是数据增强
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandShiftIntensityd,
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot

# 1. cuDNN 自动调优
torch.backends.cudnn.benchmark = True

class UNetWithDropout(UNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 在瓶颈层后插入 Dropout
        self.bottleneck_dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        x = super().forward(x)
        x = self.bottleneck_dropout(x)
        return x

class BaselineUNetSegmenter:
    def __init__(self, save_dir: str, lr: float = 1e-4,
                 val_split: float = 0.2, in_channels: int = 4, 
                patience: int = 10, 
                lr_factor: float = 0.5,
                lr_patience: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.val_split = val_split
        self.in_channels = in_channels
        self.es_patience = patience            # 用于 early stopping
        self.lr_factor = lr_factor            # ReduceLROnPlateau 因子
        self.lr_patience = lr_patience        # ReduceLROnPlateau 耐心

        # 标签映射
        self.LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
        self.num_classes = len(self.LABEL_MAP)

        # 模型、优化器、损失、度量
        self.model = None
        self.optimizer = None
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )

    def _build_model(self):
        self.model = UNetWithDropout(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2,
            norm="instance",
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='max',
            factor=self.lr_factor,
            patience=self.lr_patience,
            verbose=True
        )
        self.scaler = torch.cuda.amp.GradScaler()

    def _get_dataloader(self, data_list, batch_size, shuffle, num_workers):
        # 通用预处理
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Lambdad(keys="label", func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                                 b_min=0.0, b_max=1.0, clip=True),
            DivisiblePadd(keys=["image", "label"], k=8),
        ]
        # 仅在训练时添加随机增强
        if shuffle:
            transforms += [
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.10),
            ]
        # 转张量
        transforms.append(ToTensord(keys=["image", "label"]))

        #ds = CacheDataset(data=data_list, transform=transforms)
        #cache_rate = 1.0 if not shuffle else 0.35

        cache_rate = 0.2
        
        ds = CacheDataset(
            data=data_list,
            transform=transforms,
            cache_rate=cache_rate,     # 0 表示不缓存
        )

        
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,      # 或更多，视 CPU 核心数而定
            pin_memory=True,
            collate_fn=pad_list_data_collate
        )

        print("Number of Workers:", loader.num_workers)

        return loader

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

        
        print(f"使用 val_split={self.val_split:.2f} 划分")
        print("原始 training 样本数:", len(examples))
        print("train_exs 样本数:", len(train_exs))
        print("val_exs   样本数:", len(val_exs))
        print("batch_size:", batch_size)
        
        train_loader = self._get_dataloader(train_data, batch_size, True, num_workers)
        val_loader   = self._get_dataloader(val_data,   batch_size, False, num_workers)
        
        print(f"Samples / Batches →")
        print(f"  train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"  val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")



        
        best_val = 0.0
        no_improve = 0
        for ep in range(1, epochs + 1):
            # 训练
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                lbl  = batch["label"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    loss  = self.loss_fn(preds, lbl)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            # 验证
            self.model.eval()
            self.val_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device, non_blocking=True)
                    lbl  = batch["label"].to(self.device, non_blocking=True)
                    preds = sliding_window_inference(
                        imgs, roi_size=(128, 128, 128), sw_batch_size=1,
                        predictor=self.model, overlap=0.5
                    )
                    pred_lbl = torch.argmax(preds, dim=1, keepdim=True)
                    oh_pred = one_hot(pred_lbl, num_classes=self.num_classes)
                    oh_true = one_hot(lbl,      num_classes=self.num_classes)
                    self.val_metric(y_pred=oh_pred, y=oh_true)
                val_dice = self.val_metric.aggregate().item()
                self.val_metric.reset()

            print(f"[Epoch {ep}/{epochs}] "
                  f"train loss: {total_loss/len(train_loader):.4f} | "
                  f"val Dice: {val_dice:.4f}")
            
            self.scheduler.step(val_dice)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f" → lr now: {current_lr:.2e}")

            if val_dice > best_val:
                best_val = val_dice
                no_improve = 0
                torch.save(self.model.state_dict(), str(self.save_dir / "best_model.pt"))
                print(f"  ➞ New best at epoch {ep}: val Dice {val_dice:.4f}, saving model.")

            else:
                no_improve += 1
                if no_improve >= self.es_patience:
                    print(f"Early stopping at epoch {ep}, no improvement for {self.es_patience} epochs.")
                    break

        print(f"训练完成，最佳 val Dice={best_val:.4f}")

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model 未初始化，请先调用 fit_from_nnunet()。")
        self.model.eval()
        with torch.no_grad():
            return self.model(img.to(self.device))

    def evaluate_from_nnunet(self, nnunet_raw: str, task_id: str,
                             batch_size: int = 1, num_workers: int = 4) -> dict:
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        test_list = info.get("test", info.get("testing", []))

        # 测试只做预处理
        test_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Lambdad(keys="label", func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                                 b_min=0.0, b_max=1.0, clip=True),
            DivisiblePadd(keys=["image", "label"], k=8),
            ToTensord(keys=["image", "label"]),
        ]

        data_list = [
            {"image": [str(ds / "imagesTs" / fn) for fn in ex["image"]],
             "label": str(ds / "labelsTs" / ex["label"])}
            for ex in test_list
        ]
        loader = DataLoader(
            CacheDataset(data=data_list, transform=test_transforms),
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
                lbl  = batch["label"].to(self.device)
                preds = sliding_window_inference(
                    imgs, roi_size=(128, 128, 128), sw_batch_size=1,
                    predictor=self.model, overlap=0.5
                )
                pred_lbl = torch.argmax(preds, dim=1)

                # 二分类指标
                y_pred_bin = (pred_lbl > 0)
                y_true_bin = (lbl > 0)
                tp += int((y_pred_bin & y_true_bin).sum())
                tn += int((~y_pred_bin & ~y_true_bin).sum())
                fp += int((y_pred_bin & ~y_true_bin).sum())
                fn += int((~y_pred_bin & y_true_bin).sum())

                # 一热编码计算 Dice & HD95
                pred_oh = one_hot(pred_lbl.unsqueeze(1), num_classes=self.num_classes)
                true_oh = one_hot(lbl,                 num_classes=self.num_classes)

                self.test_dice_metric.reset()
                self.test_dice_metric(y_pred=pred_oh, y=true_oh)
                per_case_dice.append(self.test_dice_metric.aggregate().item())

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

        print(f"Test Results → Dice: {mean_dice:.4f} ± {dice_var:.6f} "
              f"(95% CI [{dice_ci_l:.3f}, {dice_ci_u:.3f}])")
        print(f"             HD95: {mean_hd95:.2f} ± {hd95_var:.4f} "
              f"(95% CI [{hd95_ci_l:.2f}, {hd95_ci_u:.2f}])")
        print(f"       Sensitivity: {sensitivity:.4f} ± {sens_var:.6f} "
              f"(95% CI [{sens_ci_l:.3f}, {sens_ci_u:.3f}])")
        print(f"       Specificity: {specificity:.4f} ± {spec_var:.6f} "
              f"(95% CI [{spec_ci_l:.3f}, {spec_ci_u:.3f}])")

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
