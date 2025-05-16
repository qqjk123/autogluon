# v3
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
from monai.inferers import sliding_window_inference


def center_crop(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_shape = src.shape[2:]
    tgt_shape = target.shape[2:]
    diffs = [s - t for s, t in zip(src_shape, tgt_shape)]
    crops = [d // 2 for d in diffs]
    return src[
        :, :,
        crops[0]:crops[0] + tgt_shape[0],
        crops[1]:crops[1] + tgt_shape[1],
        crops[2]:crops[2] + tgt_shape[2],
    ]

class RemapLabels:
    def __init__(self, label_map: dict):
        self.label_map = label_map

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=np.int32)
        for orig_val, new_val in self.label_map.items():
            out[x == orig_val] = new_val
        return out

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
        self.channels = [base_channels * m for m in channel_multipliers]
        self.num_levels = num_levels
        self.num_images = num_images
        self.dropout = nn.Dropout3d(p=dropout_prob)

        # encoders per modality
        self.image_encoders = nn.ModuleList()
        for _ in range(num_images):
            layers = []
            layers.append(Convolution(
                spatial_dims, in_channels, self.channels[0],
                strides=1, kernel_size=3,
                act=("RELU", {"inplace": True}),
                norm=("GROUP", {"num_groups": 8}),
            ))
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

        # bottleneck merging all modalities
        self.bottleneck = Convolution(
            spatial_dims,
            self.channels[num_levels - 1] * num_images,
            self.channels[-1],
            strides=1, kernel_size=3,
            act=("RELU", {"inplace": True}),
            norm=("GROUP", {"num_groups": 8}),
        )

        # decoder path
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
            for _ in range((len(branch) - 1) // 2):
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
            x = self.decoders[i](torch.cat([x] + skips, dim=1))
            x = self.dropout(x)

        return self.final_conv(x)

class UNetSeg:
    def __init__(self, save_dir: str, lr: float = 1e-4, val_split: float = 0.2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.val_split = val_split

        # metrics
        self.train_metric     = DiceMetric(include_background=False, reduction="mean")
        self.val_metric       = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric      = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        self.test_dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.num_classes = 0

    def _build_model(self, num_images: int, num_classes: int):
        self.model = FlexibleMONAI_UNet_MultiImage_SingleDecoder(
            spatial_dims=3, in_channels=1,
            out_channels=num_classes, num_images=num_images,
            base_channels=24, num_levels=4, dropout_prob=0.2
        ).to(self.device)
        self.loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.num_classes = num_classes

    def sw_infer(self, inputs_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Sliding-window inference for multi-modality inputs.
        inputs_list: list of tensors [B,1,D,H,W]
        returns: logits [B, C, D, H, W]
        """
        # combine modalities into one tensor along channel dim
        vol = torch.cat(inputs_list, dim=1)
        # predictor splits back into modality list and adds channel dim
        def _predict(x):
            # x: Tensor [B, C, D, H, W]
            x_list = [xi.unsqueeze(1) for xi in torch.unbind(x, dim=1)]
            return self.model(x_list)
        return sliding_window_inference(
            vol,
            roi_size=(64,64,64),
            sw_batch_size=1,
            predictor=_predict,
            overlap=0.5,
        )

    def fit_from_nnunet(self, nnunet_raw: str, task_id: str,
                        epochs: int = 50, batch_size: int = 1,
                        cache_rate: float = 0.2, num_workers: int = 4):
        # prepare dataset split
        ds = Path(nnunet_raw)/f"Dataset{task_id}"
        info = json.load(open(ds/"dataset.json"))
        exs = info["training"]
        split_idx = int(len(exs)*(1-self.val_split))
        train_exs, val_exs = exs[:split_idx], exs[split_idx:]

        modalities = sorted(info["modality"].keys(), key=int)
        labels = sorted(info["labels"].keys(), key=int)
        self._build_model(len(modalities), len(labels))

        def make_loader(ex_list, shuffle):
            data = [{
                "images": [str(ds/"imagesTr"/fn) for fn in ex["image"]],
                "label":  str(ds/"labelsTr"/ex["label"])
            } for ex in ex_list]
            transforms = [
                LoadImaged(keys=["images","label"], reader=NibabelReader()),
                EnsureChannelFirstd(keys=["images","label"]),
                Spacingd(keys=["images","label"], pixdim=(1,1,1), mode=("bilinear","nearest")),
                Orientationd(keys=["images","label"], axcodes="RAS"),
                Lambdad(keys="label", func=RemapLabels({int(k):i for i,k in enumerate(labels)})),
                ScaleIntensityRanged(keys=["images"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["images","label"], source_key="images"),
                RandCropByPosNegLabeld(keys=["images","label"], label_key="label", spatial_size=(64,64,64), pos=1, neg=1, num_samples=4),
                RandFlipd(keys=["images","label"], prob=0.5, spatial_axis=[0]),
                RandRotate90d(keys=["images","label"], prob=0.5, max_k=3),
                RandZoomd(keys=["images"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
                RandGaussianNoised(keys=["images"], prob=0.2, mean=0.0, std=0.1),
                RandShiftIntensityd(keys=["images"], prob=0.2, offsets=0.10),
                ToTensord(keys=["images","label"], dtype=torch.float32),
            ]
            ds_obj = CacheDataset(data=data, transform=transforms, cache_rate=cache_rate)
            return DataLoader(ds_obj, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              collate_fn=pad_list_data_collate)

        train_loader = make_loader(train_exs, True)
        val_loader   = make_loader(val_exs, False)

        best_dice = 0.0
        for ep in range(1, epochs+1):
            # training
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = [m.unsqueeze(1).to(self.device) for m in torch.unbind(batch['images'], dim=1)]
                lbl = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(imgs)
                loss = self.loss_fn(preds, lbl)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # validation with sliding window
            self.model.eval()
            self.val_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = [m.unsqueeze(1).to(self.device) for m in torch.unbind(batch['images'], dim=1)]
                    preds = self.sw_infer(imgs)
                    pred_lbl = torch.argmax(preds, dim=1)
                    self.val_metric(
                        y_pred=one_hot(pred_lbl.unsqueeze(1), self.num_classes),
                        y=one_hot(batch['label'].unsqueeze(1).to(self.device), self.num_classes)
                    )
            val_dice = self.val_metric.aggregate().item()
            self.val_metric.reset()
            print(f"Epoch {ep}/{epochs} - train_loss: {total_loss/len(train_loader):.4f}, val_dice: {val_dice:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), str(self.save_dir/"best_model.pt"))
        print(f"Training complete. Best val Dice: {best_dice:.4f}")

    def evaluate_from_nnunet(self, nnunet_raw: str, task_id: str,
                             batch_size: int = 1, num_workers: int = 4) -> dict:
        ds = Path(nnunet_raw)/f"Dataset{task_id}"
        info = json.load(open(ds/"dataset.json"))
        test_exs = info.get("test", info.get("testing", []))

        transforms = [
            LoadImaged(keys=["images","label"], reader=NibabelReader()),
            EnsureChannelFirstd(keys=["images","label"]),
            Spacingd(keys=["images","label"], pixdim=(1,1,1), mode=("bilinear","nearest")),
            Orientationd(keys=["images","label"], axcodes="RAS"),
            Lambdad(keys="label", func=RemapLabels({int(k):i for i,k in enumerate(sorted(info["labels"].keys(), key=int))})),
            ScaleIntensityRanged(keys=["images"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=["images","label"], dtype=torch.float32),
        ]
        data = [{
            "images": [str(ds/"imagesTs"/fn) for fn in ex["image"]],
            "label":  str(ds/"labelsTs"/ex["label"])
        } for ex in test_exs]
        loader = DataLoader(CacheDataset(data=data, transform=transforms, cache_rate=0.0),
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            collate_fn=pad_list_data_collate)

        per_case_dice, per_case_hd95 = [], []
        tp = tn = fp = fn = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = [m.unsqueeze(1).to(self.device) for m in torch.unbind(batch['images'], dim=1)]
                preds = self.sw_infer(imgs)
                pred_lbl = torch.argmax(preds, dim=1)
                ypb = (pred_lbl > 0); ytb = (batch['label'].to(self.device) > 0)
                tp += int((ypb & ytb).sum()); tn += int((~ypb & ~ytb).sum())
                fp += int((ypb & ~ytb).sum()); fn += int((~ypb & ytb).sum())
                self.test_dice_metric.reset()
                                # compute one-hot encoded predictions and ground truth
                pred_oh = one_hot(pred_lbl.unsqueeze(1), self.num_classes)
                true_oh = one_hot(batch['label'].to(self.device), self.num_classes)

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

