import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np

from monai.networks.blocks import PatchEmbeddingBlock
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    Lambdad,
    ScaleIntensityRanged,
    DivisiblePadd,
    ToTensord,
    ResizeWithPadOrCropd,
    # Data augmentation transforms
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandShiftIntensityd,
)
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from monai.networks.blocks import PatchEmbed, TransformerBlock


# Enable cuDNN autotuning
torch.backends.cudnn.benchmark = True

class RemapLabels:
    """
    A picklable label-remapping transform.
    """
    def __init__(self, label_map):
        self.label_map = label_map

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        # vectorize over the numpy array
        return np.vectorize(self.label_map.get)(arr)


class ViTSegmentationEncoder(nn.Module):
    """Vision Transformer encoder for 3D medical image segmentation."""
    
    def __init__(
        self,
        in_channels: int = 4,
        img_size: tuple = (128, 128, 128),
        patch_size: tuple = (16, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Patch embedding
        
        self.patch_embedding = PatchEmbed(
            patch_size=patch_size,     # e.g. (8,8,8)
            in_chans=in_channels,      # e.g. 4
            embed_dim=hidden_size,     # e.g. 512
            norm_layer=None,           # or nn.LayerNorm if you want a post-proj norm
            spatial_dims=3,            # because this is 3D data
        )

        
        # Calculate number of patches
        self.patch_dims = [img_size[i] // patch_size[i] for i in range(3)]
        self.num_patches = np.prod(self.patch_dims)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Position embedding
        if pos_embed == "conv":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
            # Initialize position embeddings
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2.0, b=2.0)
            
        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                qkv_bias=True
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # For reshaping back to volumetric feature maps
        self.patch_dims = self.patch_dims
        self.hidden_size = hidden_size
        
    def forward(self, x):
        # Get patches
        x = self.patch_embedding(x)
        
        # Reshape for transformer blocks
        B, C, *patch_dims = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # B, N, C
        
        # Prepend class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Remove class token for segmentation
        tokens = x[:, 1:, :]
        
        # Reshape back to volumetric feature map
        x = tokens.permute(0, 2, 1).reshape(B, C, *self.patch_dims)
        
        return x

class UpBlock(nn.Module):
    """Upsampling block for decoder."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x):
        x = self.upconv(x)
        x = self.conv(x)
        return x

class ViTSegmentationModel(nn.Module):
    """Vision Transformer for 3D medical image segmentation with decoder."""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        img_size: tuple = (128, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        # Calculate patch size based on feature size
        patch_size = (img_size[0] // feature_size, img_size[1] // feature_size, img_size[2] // feature_size)
        
        # ViT encoder
        self.encoder = ViTSegmentationEncoder(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
        )
        
        # Decoder pathway with upsampling
        self.decoder1 = UpBlock(hidden_size, hidden_size // 2)
        self.decoder2 = UpBlock(hidden_size // 2, hidden_size // 4)
        self.decoder3 = UpBlock(hidden_size // 4, hidden_size // 8)
        
        # Final convolution to get the desired number of output channels
        self.final_conv = nn.Conv3d(hidden_size // 8, out_channels, kernel_size=1)
        
        # Spatial dropout for regularization
        self.dropout = nn.Dropout3d(p=0.2)
        
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Apply dropout for regularization
        encoded = self.dropout(encoded)
        
        # Decoder
        x = self.decoder1(encoded)
        x = self.decoder2(x)
        x = self.decoder3(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class ViTSegmenter:
    def __init__(self, save_dir: str, lr: float = 1e-4,
                 val_split: float = 0.2, in_channels: int = 4):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.val_split = val_split
        self.in_channels = in_channels

        # Label mapping
        self.LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
        self.num_classes = len(self.LABEL_MAP)

        # Model, optimizer, loss, metrics
        self.model = None
        self.optimizer = None
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.val_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.test_hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )

    def _build_model(self):
        self.model = ViTSegmentationModel(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            img_size=(128, 128, 128),  # Adaptable based on data
            feature_size=16,
            hidden_size=512,  # Reduced from 768 for memory efficiency
            mlp_dim=2048,     # Reduced from 3072 for memory efficiency
            num_layers=8,     # Reduced from 12 for memory efficiency
            num_heads=8,      # Reduced from 12 for memory efficiency
            dropout_rate=0.1,
        ).to(self.device)
        
        # Initialize AdamW optimizer which is better for transformers
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=0.01  # L2 regularization, common for transformers
        )
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def _get_dataloader(self, data_list, batch_size, shuffle, num_workers):
        # Common preprocessing
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ResizeWithPadOrCropd(keys=["image","label"], spatial_size=(128,128,128)),
            Lambdad(keys="label", func=RemapLabels(self.LABEL_MAP)),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                                 b_min=0.0, b_max=1.0, clip=True),
            DivisiblePadd(keys=["image", "label"], k=16),  # Changed to 16 for ViT compatibility
        ]
        # Only add random augmentation during training
        if shuffle:
            transforms += [
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.9, max_zoom=1.1),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.10),
            ]
        # Convert to tensor
        transforms.append(ToTensord(keys=["image", "label"]))

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
            # 1) 训练步骤
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(self.device, non_blocking=True)
                lbls = batch["label"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    loss = self.loss_fn(preds, lbls)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
            self.scheduler.step()

            # 2) 验证步骤：累积全量标签并计算 Dice
            all_true = set()
            all_pred = set()
            self.val_metric.reset()
            self.model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(self.device, non_blocking=True)
                    lbls = batch["label"].to(self.device, non_blocking=True)
                    preds = sliding_window_inference(
                        imgs,
                        roi_size=(128, 128, 128),
                        sw_batch_size=1,
                        predictor=self.model,
                        overlap=0.5,
                    )
                    pred_lbl = torch.argmax(preds, dim=1)

                    # 累积所有 batch 的真实和预测标签
                    all_true.update(torch.unique(lbls).cpu().numpy().tolist())
                    all_pred.update(torch.unique(pred_lbl).cpu().numpy().tolist())

                    # 计算 Dice 指标
                    oh_pred = one_hot(pred_lbl.unsqueeze(1), num_classes=self.num_classes)
                    oh_true = one_hot(lbls, num_classes=self.num_classes)
                    self.val_metric(y_pred=oh_pred, y=oh_true)

                # 输出全量标签覆盖情况
                print(f"[Epoch {ep}] 验证集真实标签：{sorted(all_true)}")
                print(f"[Epoch {ep}] 验证集预测标签：{sorted(all_pred)}")

                val_dice = self.val_metric.aggregate().item()
                self.val_metric.reset()

            # 3) 打印并保存最优模型
            lr = self.scheduler.get_last_lr()[0]
            print(f"[Epoch {ep}/{epochs}] "
                  f"train loss: {total_loss/len(train_loader):.4f} | "
                  f"val Dice: {val_dice:.4f} | lr: {lr:.6f}")

            if val_dice > best_val:
                best_val = val_dice
                torch.save(self.model.state_dict(), str(self.save_dir / "best_model.pt"))
                model_meta = {
                    "architecture": "ViTSegmentationModel",
                    "in_channels": self.in_channels,
                    "num_classes": self.num_classes,
                    "best_val_dice": best_val,
                    "epoch": ep
                }
                with open(self.save_dir / "model_meta.json", "w") as f:
                    json.dump(model_meta, f, indent=2)

        print(f"Training completed, best val Dice={best_val:.4f}")

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized, please call fit_from_nnunet() first.")
        self.model.eval()
        with torch.no_grad():
            return self.model(img.to(self.device))

    def evaluate_from_nnunet(self, nnunet_raw: str, task_id: str,
                             batch_size: int = 1, num_workers: int = 4) -> dict:
        ds = Path(nnunet_raw) / f"Dataset{task_id}"
        info = json.load(open(ds / "dataset.json"))
        test_list = info.get("test", info.get("testing", []))

        # Test preprocessing only
        test_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ResizeWithPadOrCropd(keys=["image","label"], spatial_size=(128,128,128)),
            Lambdad(keys="label", func=RemapLabels(self.LABEL_MAP)),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                                 b_min=0.0, b_max=1.0, clip=True),
            DivisiblePadd(keys=["image", "label"], k=16),  # Changed to 16 for ViT compatibility
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
                lbl = batch["label"].to(self.device)
                preds = sliding_window_inference(
                    imgs, roi_size=(128, 128, 128), sw_batch_size=1,
                    predictor=self.model, overlap=0.5
                )
                pred_lbl = torch.argmax(preds, dim=1)

                # Binary classification metrics
                y_pred_bin = (pred_lbl > 0)
                y_true_bin = (lbl > 0)
                tp += int((y_pred_bin & y_true_bin).sum())
                tn += int((~y_pred_bin & ~y_true_bin).sum())
                fp += int((y_pred_bin & ~y_true_bin).sum())
                fn += int((~y_pred_bin & y_true_bin).sum())

                # One-hot encoding for Dice & HD95
                pred_oh = one_hot(pred_lbl.unsqueeze(1), num_classes=self.num_classes)
                true_oh = one_hot(lbl, num_classes=self.num_classes)

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

