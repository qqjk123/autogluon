# Version 2.0: 增强、Dropout、网络加宽
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

from monai.data import CacheDataset, DataLoader, NibabelReader, pad_list_data_collate
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandShiftIntensityd,
    Lambdad,
    ToTensord,
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.blocks import Convolution, UpSample
from monai.networks.utils import one_hot

class RemapLabels:
    def __init__(self, label_map: dict):
        self.label_map = label_map

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.int32)
        for orig_val, new_val in self.label_map.items():
            out[x == orig_val] = new_val
        return out

def center_crop(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_shape = src.shape[2:]
    tgt_shape = target.shape[2:]
    diffs = [s - t for s, t in zip(src_shape, tgt_shape)]
    crops = [d // 2 for d in diffs]
    return src[
        :,
        :,
        crops[0]:crops[0] + tgt_shape[0],
        crops[1]:crops[1] + tgt_shape[1],
        crops[2]:crops[2] + tgt_shape[2],
    ]

class FlexibleMONAI_UNet_MultiImage_SingleDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        num_images=4,
        base_channels=24,
        num_levels=4,
        channel_multipliers=None,
        up_mode="deconv",
        dropout_prob=0.2,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8, 16]
        assert len(channel_multipliers) == num_levels + 1

        self.channels = [base_channels * m for m in channel_multipliers]
        self.num_levels = num_levels
        self.num_images = num_images
        self.dropout = nn.Dropout3d(p=dropout_prob)

        # encoders
        self.image_encoders = nn.ModuleList()
        for _ in range(num_images):
            layers = []
            # 首层
            layers.append(Convolution(
                spatial_dims, in_channels, self.channels[0],
                strides=1, kernel_size=3,
                act=("RELU", {"inplace": True}),
                norm=("GROUP", {"num_groups": 8}),
            ))
            # 后续级别
            for i in range(1, num_levels):
                layers.append(Convolution(
                    spatial_dims, self.channels[i-1], self.channels[i],
                    strides=2, kernel_size=3,
                    act=("RELU", {"inplace": True}),
                    norm=("GROUP", {"num_groups": 8}),
                ))
                layers.append(Convolution(
                    spatial_dims, self.channels[i], self.channels[i],
                    strides=1, kernel_size=3,
                    act=("RELU", {"inplace": True}),
                    norm=("GROUP", {"num_groups": 8}),
                ))
            self.image_encoders.append(nn.ModuleList(layers))

        # bottleneck
        self.bottleneck = Convolution(
            spatial_dims,
            self.channels[num_levels - 1] * num_images,
            self.channels[-1],
            strides=1,
            kernel_size=3,
            act=("RELU", {"inplace": True}),
            norm=("GROUP", {"num_groups": 8}),
        )

        # decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1):
            in_ch = self.channels[-1] if i == 0 else self.channels[num_levels - 1 - i]
            target_ch = self.channels[num_levels - 2 - i]
            self.upsamples.append(UpSample(
                spatial_dims, in_ch, target_ch,
                scale_factor=2, mode=up_mode,
            ))
            self.decoders.append(Convolution(
                spatial_dims,
                in_channels=target_ch * (num_images + 1),
                out_channels=target_ch,
                strides=1, kernel_size=3,
                act=("RELU", {"inplace": True}),
                norm=("GROUP", {"num_groups": 8}),
            ))

        # final conv
        self.final_conv = Convolution(
            spatial_dims,
            in_channels=self.channels[0],
            out_channels=out_channels,
            strides=1, kernel_size=1,
            act=None, norm=None,
        )

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        skips_all, feats = [], []
        for branch, x in zip(self.image_encoders, x_list):
            skip_feats = []
            out = branch[0](x)
            skip_feats.append(out)
            idx = 1
            num_blocks = (len(branch) - 1) // 2
            for _ in range(num_blocks):
                out = branch[idx](out); idx += 1
                out = branch[idx](out); idx += 1
                skip_feats.append(out)
            skips_all.append(skip_feats)
            feats.append(skip_feats[-1])

        x = torch.cat(feats, dim=1)
        x = self.bottleneck(x)
        x = self.dropout(x)

        for i in range(self.num_levels - 1):
            x = self.upsamples[i](x)
            skips = []
            for skip_feats in skips_all:
                skip = skip_feats[self.num_levels - 2 - i]
                if x.shape[2:] != skip.shape[2:]:
                    if x.shape[2] < skip.shape[2]:
                        skip = center_crop(skip, x)
                    else:
                        x = center_crop(x, skip)
                skips.append(skip)
            x = self.decoders[i](torch.cat([x, *skips], dim=1))
            x = self.dropout(x)

        return self.final_conv(x)

class UNetSeg:
    def __init__(self, save_dir: str, lr: float = 1e-4, val_split: float = 0.2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.val_split = val_split

        self.loss_fn = None
        self.optimizer = None

        self.train_metric     = DiceMetric(include_background=False, reduction="mean")
        self.val_metric       = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric      = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        self.test_dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        self.model = None

    def _build_model(self, num_images: int, num_classes: int):
        self.model = FlexibleMONAI_UNet_MultiImage_SingleDecoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            num_images=num_images,
            base_channels=24,
            num_levels=4,
            dropout_prob=0.2,
        ).to(self.device)
        self.loss_fn  = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def load(self, checkpoint_path: str, nnunet_raw: str = None, task_id: str = None):
        if self.model is None:
            assert nnunet_raw and task_id, "首次 load 需提供 nnunet_raw 和 task_id"
            ds   = Path(nnunet_raw) / f"Dataset{task_id}"
            info = json.load(open(ds / "dataset.json"))
            num_images  = len(info["modality"])
            num_classes = len(info["labels"])
            self._build_model(num_images, num_classes)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def fit_from_nnunet(
        self,
        nnunet_raw: str,
        task_id: str,
        epochs: int = 50,
        batch_size: int = 1,
        cache_rate: float = 0.2,
        num_workers: int = 4
    ):
        ds   = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        ex_list = info["training"]

        split_idx = int(len(ex_list) * (1 - self.val_split))
        train_exs = ex_list[:split_idx]
        val_exs   = ex_list[split_idx:]

        modalities = sorted(info["modality"].keys(), key=int)
        labels     = sorted(info["labels"].keys(), key=int)
        label_map  = {int(o): i for i, o in enumerate(labels)}
        self._build_model(len(modalities), len(labels))

        def make_loader(exs, shuffle):
            data = [{
                "images": [str(ds/"imagesTr"/fn) for fn in ex["image"]],
                "label":  str(ds/"labelsTr"/ex["label"])
            } for ex in exs]
            transforms = [
                LoadImaged(keys=["images","label"], reader=NibabelReader()),
                EnsureChannelFirstd(keys=["images","label"]),
                Spacingd(keys=["images","label"], pixdim=(1,1,1), mode=("bilinear","nearest")),
                Orientationd(keys=["images","label"], axcodes="RAS"),
                Lambdad(keys="label", func=RemapLabels(label_map)),
                ScaleIntensityRanged(keys=["images"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["images","label"], source_key="images"),
                RandCropByPosNegLabeld(keys=["images","label"], label_key="label", spatial_size=(64,64,64), pos=1, neg=1, num_samples=4),
                RandFlipd(keys=["images","label"], prob=0.5, spatial_axis=[0]),
                RandRotate90d(keys=["images","label"], prob=0.5, max_k=3),
                RandZoomd(keys=["images","label"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
                RandGaussianNoised(keys=["images"], prob=0.2, mean=0.0, std=0.1),
                RandShiftIntensityd(keys=["images"], prob=0.2, offsets=0.10),
                ToTensord(keys=["images","label"]),
            ]
            ds_obj = CacheDataset(data=data, transform=transforms, cache_rate=cache_rate)
            return DataLoader(ds_obj, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available())

        train_loader = make_loader(train_exs, shuffle=True)
        val_loader   = make_loader(val_exs,   shuffle=False)

        best_val_dice = 0.0
        for ep in range(1, epochs+1):
            # 训练
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = [m.unsqueeze(1).to(self.device) for m in torch.unbind(batch['images'], dim=1)]
                lbl  = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(imgs)
                loss  = self.loss_fn(preds, lbl)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # 验证
            self.model.eval()
            self.val_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = [m.unsqueeze(1).to(self.device) for m in torch.unbind(batch['images'], dim=1)]
                    lbl  = batch['label'].to(self.device)
                    preds = self.model(imgs)
                    pred_lbl = torch.argmax(preds, dim=1)
                    pred_oh  = one_hot(pred_lbl.unsqueeze(1), preds.shape[1])
                    true_oh  = one_hot(lbl.unsqueeze(1), preds.shape[1])
                    self.val_metric(y_pred=pred_oh, y=true_oh)
            val_dice = self.val_metric.aggregate().item()
            self.val_metric.reset()

            print(f"Epoch {ep}/{epochs} - train_loss: {total_loss/len(train_loader):.4f}, val_dice: {val_dice:.4f}")
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(self.model.state_dict(), str(self.save_dir/"best_model.pt"))

        print(f"Training complete. Best val Dice: {best_val_dice:.4f}")

        final_metrics = self.evaluate_from_nnunet(nnunet_raw, task_id, batch_size, num_workers)
        print(
            f"Final Test → "
            f"DSC: {final_metrics['dice']:.4f}, "
            f"HD95: {final_metrics['hd95']:.4f}, "
            f"Sensitivity: {final_metrics['sensitivity']:.4f}, "
            f"Specificity: {final_metrics['specificity']:.4f}"
        )

    def predict(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call load() or fit_from_nnunet() first.")
        self.model.eval()
        with torch.no_grad():
            return self.model([x.to(self.device) for x in x_list])

    def evaluate_from_nnunet(
        self, nnunet_raw: str, task_id: str,
        batch_size: int = 1, num_workers: int = 4
    ) -> dict:
        import numpy as np
        from monai.data import pad_list_data_collate

        ds   = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds/"dataset.json"))
        test_list = info.get("test", info.get("testing", []))

        test_transforms = [
            LoadImaged(keys=["images","label"], reader=NibabelReader()),
            EnsureChannelFirstd(keys=["images","label"]),
            Spacingd(keys=["images","label"], pixdim=(1,1,1), mode=("bilinear","nearest")),
            Orientationd(keys=["images","label"], axcodes="RAS"),
            Lambdad(
                keys="label",
                func=RemapLabels({int(k): i for i, k in enumerate(sorted(info["labels"].keys(), key=int))})
            ),
            ScaleIntensityRanged(
                keys=["images"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["images","label"], source_key="images"),
            ToTensord(keys=["images","label"]),
        ]

        data_list = [
            {
                "images": [str(ds/"imagesTs"/fn) for fn in ex["image"]],
                "label":  str(ds/"labelsTs"/ex["label"])
            }
            for ex in test_list
        ]

        loader = DataLoader(
            CacheDataset(data=data_list, transform=test_transforms, cache_rate=0.0),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=pad_list_data_collate,
        )

        per_case_dice, per_case_hd95 = [], []
        tp = tn = fp = fn = 0
        self.test_hd95_metric.reset()
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = [m.unsqueeze(1).to(self.device)
                        for m in torch.unbind(batch["images"], dim=1)]
                lbl = batch["label"].to(self.device)
                preds = self.model(imgs)
                pred_lbl = torch.argmax(preds, dim=1)

                y_pred_bin = (pred_lbl > 0)
                y_true_bin = (lbl > 0)
                tp += int((y_pred_bin & y_true_bin).sum())
                tn += int((~y_pred_bin & ~y_true_bin).sum())
                fp += int((y_pred_bin & ~y_true_bin).sum())
                fn += int((~y_pred_bin & y_true_bin).sum())

                pred_oh = one_hot(pred_lbl.unsqueeze(1), preds.shape[1])
                true_oh = one_hot(lbl, preds.shape[1])
                self.test_dice_metric.reset()
                self.test_dice_metric(y_pred=pred_oh, y=true_oh)
                per_case_dice.extend([self.test_dice_metric.aggregate().item()]*batch_size)

                self.test_hd95_metric(y_pred=pred_oh, y=true_oh)
                per_case_hd95.extend([self.test_hd95_metric.aggregate().item()]*batch_size)
                self.test_hd95_metric.reset()

        mean_dice = float(np.mean(per_case_dice))
        mean_hd95 = float(np.mean(per_case_hd95))
        mean_sens = tp / (tp + fn) if (tp + fn) else 0.0
        mean_spec = tn / (tn + fp) if (tn + fp) else 0.0

        return {
            "dice": mean_dice,
            "hd95": mean_hd95,
            "sensitivity": mean_sens,
            "specificity": mean_spec,
        }
