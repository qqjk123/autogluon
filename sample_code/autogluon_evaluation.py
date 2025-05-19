#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
from PIL import Image
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from autogluon.multimodal import MultiModalPredictor
from pathlib import Path

def stack_slices(df, preds, axis=2):
    """
    将同一 case 的所有 2D 切片按 z 顺序重组成 3D 体
    假设 slice 文件名里包含 '_z###' 作为层索引。
    """
    volumes = {}
    for row, mask2d in zip(df.itertuples(), preds):
        stem = Path(row.image).stem  # e.g. "train_BraTS-MET-00004-000_z12"
        try:
            case_id, z_str = stem.rsplit('_z', 1)
        except ValueError:
            raise ValueError(f"Filename {stem} does not match expected pattern '..._z###'")
        z = int(z_str)

        shape2d = mask2d.shape[-2:]  # 兼容 (H,W) 或 (1,H,W)
        if case_id not in volumes:
            volumes[case_id] = {'slices': {}, 'shape2d': shape2d}
        volumes[case_id]['slices'][z] = mask2d

    out = {}
    for cid, info in volumes.items():
        nz = max(info['slices'].keys()) + 1
        h, w = info['shape2d']
        vol = np.zeros((1, nz, h, w), dtype=np.int64)  # shape: (C=1, D, H, W)
        for z_idx, sl in info['slices'].items():
            # 如果 sl.ndim == 3，则取第一个通道
            sl2d = sl[0] if sl.ndim == 3 else sl
            vol[0, z_idx] = sl2d
        out[cid] = vol
    return out

def main():
    # 1. Load predictor & test csv
    predictor = MultiModalPredictor.load("automm_dataset001")
    df_test = pd.read_csv("/workspace/autogluon_original/test.csv")

    # 2. Predict & load GT
    preds2d = predictor.predict(data=df_test, save_results=False)
    gts2d = [(np.array(Image.open(p)) > 0).astype(np.int64) for p in df_test["label"]]

    # 3. 重组 3D
    pred_vols = stack_slices(df_test, preds2d)
    gt_vols   = stack_slices(df_test, gts2d)

    # 4. 设置 MONAI 的 3D 指标
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)

    dice_list, hd_list = [], []
    for case_id in pred_vols:
        p = torch.from_numpy(pred_vols[case_id])  # (C=1, D, H, W)
        g = torch.from_numpy(gt_vols[case_id])

        dice_metric(p, g)
        hd_metric(p, g)

        # unpack tuple: (value_tensor, not_nans_count)
        dice_val, _ = dice_metric.aggregate()
        hd_val, _   = hd_metric.aggregate()

        dice_list.append(dice_val.item())
        hd_list.append(hd_val.item())

        dice_metric.reset()
        hd_metric.reset()

    # 5. 输出统计（和你的 UNet 基线统一格式）
    mean_dice = float(np.mean(dice_list))
    mean_hd95 = float(np.mean(hd_list))
    print(f"AutoMM 3D Case‐level → Dice: {mean_dice:.4f}, HD95: {mean_hd95:.2f} voxels")

    # 6. 你可以用同样方式调用 BaselineUNetSegmenter.evaluate_from_nnunet()
    #    它已经输出 case‐level mean ± CI，保持一致即可直接对比。

if __name__ == "__main__":
    main()
