# baseline_unet_segmenter_optimized.py

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
from monai.metrics import DiceMetric
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

        self.LABEL_MAP = {0:0, 1:1, 2:2, 3:3}
        self.num_classes = len(self.LABEL_MAP)

        self.model = None
        self.optimizer = None
        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")

    def _build_model(self):
        self.model = UNet(
            spatial_dims=3, in_channels=self.in_channels, out_channels=self.num_classes,
            channels=(16, 32, 64, 128), strides=(2,2,2), num_res_units=2
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 2. 初始化 AMP 缩放器
        self.scaler = torch.cuda.amp.GradScaler()

    def _get_dataloader(self, data_list, batch_size, shuffle, num_workers):
        transforms = [
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0),
                     mode=("bilinear","nearest")),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Lambdad(keys="label",
                    func=lambda x: np.vectorize(self.LABEL_MAP.get)(x)),
            ScaleIntensityRanged(keys=["image"],
                                 a_min=0, a_max=3000, b_min=0.0, b_max=1.0,
                                 clip=True),
            DivisiblePadd(keys=["image","label"], k=8),
            ToTensord(keys=["image","label"]),
        ]
        ds = CacheDataset(data=data_list, transform=transforms)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pad_list_data_collate
        )

    def fit_from_nnunet(self, nnunet_raw: str, task_id: str,
                        epochs: int = 50, batch_size: int = 1,
                        num_workers: int = 4):
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds/"dataset.json"))
        examples = info["training"]
        split = int(len(examples)*(1-self.val_split))
        train_exs, val_exs = examples[:split], examples[split:]
        self._build_model()

        train_data = [
            {"image":[str(ds/"imagesTr"/fn) for fn in ex["image"]],
             "label":str(ds/"labelsTr"/ex["label"])}
            for ex in train_exs
        ]
        val_data = [
            {"image":[str(ds/"imagesTr"/fn) for fn in ex["image"]],
             "label":str(ds/"labelsTr"/ex["label"])}
            for ex in val_exs
        ]
        train_loader = self._get_dataloader(train_data, batch_size, True, num_workers)
        val_loader   = self._get_dataloader(val_data,   batch_size, False, num_workers)

        best_val = 0.0
        for ep in range(1, epochs+1):
            # ———— 训练 ————
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                lbl  = batch["label"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()

                # 2.1 混合精度前向和反向
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    loss  = self.loss_fn(preds, lbl)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()

            # ———— 验证 ————
            self.model.eval()
            self.metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device, non_blocking=True)
                    lbl  = batch["label"].to(self.device, non_blocking=True)
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

            print(f"[Epoch {ep}/{epochs}] train loss: {total_loss/len(train_loader):.4f} | val Dice: {val_dice:.4f}")
            if val_dice > best_val:
                best_val = val_dice
                torch.save(self.model.state_dict(), str(self.save_dir/"best_model.pt"))

        print(f"训练完成，最佳 val Dice={best_val:.4f}")

    # predict/evaluate 部分可同样加上 AMP 和 non_blocking，略
