#!/usr/bin/env python3
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
    Lambdad,
    ScaleIntensityRanged,
    DivisiblePadd,
    ResizeWithPadOrCropd,
    ToTensord,
    # —— 以下是数据增强
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandShiftIntensityd,
)
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.networks.layers import DropPath
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from torch.optim.lr_scheduler import CosineAnnealingLR
# 1. cuDNN 自动调优
torch.backends.cudnn.benchmark = True

class BaselineUNETRSegmenter:
    def __init__(self, save_dir: str, lr: float = 1e-4,
                 val_split: float = 0.2, in_channels: int = 4):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.val_split = val_split
        self.in_channels = in_channels
        # BraTS-style 标签映射 {0,1,2,3}→{0,1,2,3}
        self.LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
        self.num_classes = len(self.LABEL_MAP)

        # placeholders，稍后在 _build_model 中初始化
        self.model = None
        self.optimizer = None
        self.scaler = None

        # 损失与度量
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )

    def _build_model(self):
        # instantiate UNETR
        self.model = UNETR(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            img_size=(128, 128, 128),
            feature_size=32,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance"
        ).to(self.device)

        # 2) Patch‐Embed 后插入 Dropout
        #    通常 patch‐embed 在 self.model.vit.patch_embed
        self.model.vit.patch_embedding = nn.Sequential(
            self.model.vit.patch_embedding,
            nn.Dropout(p=0.1)  # 丢弃 10%
        )
    
        # 3) 对每个 Transformer block 注入 DropPath（Stochastic Depth）
        #    Transformer blocks 列在 self.model.vit.blocks 列表中
        for blk in self.model.vit.blocks:
            blk.drop_path = DropPath(0.15)  # 随机丢层 15%
    
        # 4) Decoder 侧再用 3D Dropout
        self.decoder_dropout = nn.Dropout3d(p=0.2)


        # optimizer + AMP scaler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # newer API:
        self.scaler = torch.cuda.amp.GradScaler()

    def _get_dataloader(self, data_list, batch_size, shuffle, num_workers, cache_rate):
        transforms = [
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0),
                     mode=("bilinear","nearest")),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Lambdad(keys="label",
                    func=lambda x: np.vectorize(self.LABEL_MAP.get, otypes=[np.int32])(x)),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                                 b_min=0.0, b_max=1.0, clip=True),
            DivisiblePadd(keys=["image","label"], k=16),

            # —— 强制输出固定体素尺寸 128×128×128
            ResizeWithPadOrCropd(
                keys=["image","label"],
                spatial_size=(128, 128, 128)
            ),

            ToTensord(keys=["image","label"]),
        ]
        if shuffle:
            transforms += [
                RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
                RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
                RandZoomd(keys=["image","label"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                # older vs. newer API: if this errors, switch `offsets` ↔ `shift_range`
                RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.10),
            ]

        ds = CacheDataset(data=data_list, transform=transforms, cache_rate=cache_rate)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
             persistent_workers=True,
            prefetch_factor=2,
            collate_fn=pad_list_data_collate
        )

    def fit_from_nnunet(self, nnunet_raw: str, task_id: str,
                        epochs: int = 50, batch_size: int = 1,
                        num_workers: int = 4, cache_rate = 0.3):
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        examples = info["training"]
        split = int(len(examples) * (1 - self.val_split))
        train_exs, val_exs = examples[:split], examples[split:]

        self._build_model()

        train_data = [
            {
              "image": [str(ds / "imagesTr" / fn) for fn in ex["image"]],
              "label": str(ds / "labelsTr" / ex["label"])
            }
            for ex in train_exs
        ]
        val_data = [
            {
              "image": [str(ds / "imagesTr" / fn) for fn in ex["image"]],
              "label": str(ds / "labelsTr" / ex["label"])
            }
            for ex in val_exs
        ]
        train_loader = self._get_dataloader(train_data, batch_size, True, num_workers, cache_rate)
        val_loader   = self._get_dataloader(val_data,   batch_size, False, num_workers, cache_rate)

        total_steps = epochs * len(train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        
        best_val = 0.0
        for ep in range(1, epochs + 1):
            # —— 训练步骤
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                lbls = batch["label"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda"):
                    preds = self.model(imgs)
                    loss  = self.loss_fn(preds, lbls)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            # —— 验证步骤
            self.model.eval()
            self.val_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device, non_blocking=True)
                    lbls = batch["label"].to(self.device, non_blocking=True)
                    preds = sliding_window_inference(
                        imgs, roi_size=(128,128,128), sw_batch_size=2,
                        predictor=self.model, overlap=0.5
                    )
                    pred_lbl = torch.argmax(preds, dim=1, keepdim=True)
                    oh_pred = one_hot(pred_lbl, num_classes=self.num_classes)
                    oh_true = one_hot(lbls,     num_classes=self.num_classes)
                    self.val_metric(y_pred=oh_pred, y=oh_true)
                val_dice = self.val_metric.aggregate().item()
                self.val_metric.reset()

            print(f"[Epoch {ep}/{epochs}] "
                  f"train loss: {total_loss/len(train_loader):.4f} | "
                  f"val Dice: {val_dice:.4f}")

            current_lrs = self.scheduler.get_last_lr()
            # 通常也是一个长度为 1 的 list
            print(f"  lr: {current_lrs[0]:.2e}")
            
            self.scheduler.step()
            
            
            if val_dice > best_val:
                best_val = val_dice
                torch.save(self.model.state_dict(),
                           str(self.save_dir / "best_unetr.pt"))

        print(f"训练完成，最佳 val Dice = {best_val:.4f}")

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model 未初始化，请先调用 fit_from_nnunet()。")
        self.model.eval()
        with torch.no_grad():
            return self.model(img.to(self.device))

    def evaluate_from_nnunet(self, nnunet_raw: str, task_id: str,
                             batch_size: int = 1, num_workers: int = 4, cache_rate: float = 0.0) -> dict:
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        test_list = info.get("test", info.get("testing", []))

        test_transforms = [
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            Spacingd(keys=["image","label"],
                     pixdim=(1.0,1.0,1.0),
                     mode=("bilinear","nearest")),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Lambdad(keys="label",
                    func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
            ScaleIntensityRanged(keys=["image"],
                                 a_min=0, a_max=3000,
                                 b_min=0.0, b_max=1.0,
                                 clip=True),
            DivisiblePadd(keys=["image","label"], k=16),
            ResizeWithPadOrCropd(
                keys=["image","label"],
                spatial_size=(128, 128, 128)
            ),
            ToTensord(keys=["image","label"]),
        ]

        data_list = [
            {
              "image": [str(ds/"imagesTs"/fn) for fn in ex["image"]],
              "label": str(ds/"labelsTs"/ex["label"])
            } 
            for ex in test_list
        ]
        loader = DataLoader(
            CacheDataset(data=data_list, transform=test_transforms, cache_rate=cache_rate),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=pad_list_data_collate
        )

        per_case_dice = []
        per_case_hd95 = []
        per_case_sens = []
        per_case_spec = []


        self.test_hd95_metric.reset()
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(self.device)
                lbls = batch["label"].to(self.device)
    
                preds = sliding_window_inference(
                    imgs, roi_size=(128,128,128), sw_batch_size=2,
                    predictor=self.model, overlap=0.5
                )
                pred_lbl = torch.argmax(preds, dim=1)
    
                # 二分类指标 per-case
                y_pred_bin = (pred_lbl > 0)
                y_true_bin = (lbls > 0)
                tp_i = int((y_pred_bin & y_true_bin).sum())
                tn_i = int((~y_pred_bin & ~y_true_bin).sum())
                fp_i = int((y_pred_bin & ~y_true_bin).sum())
                fn_i = int((~y_pred_bin & y_true_bin).sum())
                sens_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
                spec_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
                per_case_sens.append(sens_i)
                per_case_spec.append(spec_i)
    
                # 多类 Dice & HD95 per-case
                pred_oh = one_hot(pred_lbl.unsqueeze(1), num_classes=self.num_classes)
                true_oh = one_hot(lbls,                  num_classes=self.num_classes)
    
                self.test_dice_metric.reset()
                self.test_dice_metric(y_pred=pred_oh, y=true_oh)
                per_case_dice.append(self.test_dice_metric.aggregate().item())
    
                self.test_hd95_metric.reset()
                self.test_hd95_metric(y_pred=pred_oh, y=true_oh)
                per_case_hd95.append(self.test_hd95_metric.aggregate().item())
    
        # Means and sample standard deviations
        mean_dice = float(np.mean(per_case_dice))
        std_dice  = float(np.std(per_case_dice, ddof=1))
        mean_hd95 = float(np.mean(per_case_hd95))
        std_hd95  = float(np.std(per_case_hd95, ddof=1))
        mean_sens = float(np.mean(per_case_sens))
        std_sens  = float(np.std(per_case_sens, ddof=1))
        mean_spec = float(np.mean(per_case_spec))
        std_spec  = float(np.std(per_case_spec, ddof=1))
    
        print(f"Dice:        {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"HD95:        {mean_hd95:.2f} ± {std_hd95:.2f}")
        print(f"Sensitivity: {mean_sens:.4f} ± {std_sens:.4f}")
        print(f"Specificity: {mean_spec:.4f} ± {std_spec:.4f}")
    
        return {
            "dice": mean_dice,
            "dice_std": std_dice,
            "hd95": mean_hd95,
            "hd95_std": std_hd95,
            "sensitivity": mean_sens,
            "sens_std": std_sens,
            "specificity": mean_spec,
            "spec_std": std_spec,
        }
