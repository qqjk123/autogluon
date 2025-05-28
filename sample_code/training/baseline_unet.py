#!/usr/bin/env python3
import os
from pathlib import Path

# 如果 BaselineUNetSegmenter 定义在同目录下的 baseline_unet.py

from autogluon.multimodal.learners import BaselineUNetSegmenter




# —————————————————————————————————————————————————————————————————————————————
# 直接在这里设置你的训练参数，无需 argparse
nnunet_raw  = "/workspace/nnUnet_raw"         # 父目录，里面有 Dataset001_BraTS2023_SEG 文件夹
task_id     = "002_BraTS2023_MEN"             # 不要写 'Dataset' 前缀，也不要写 'dataset.json'
save_dir    = "./checkpoint/checkpoints_012_v2"
epochs      = 50
batch_size  = 6
num_workers = 16
lr          = 1e-4
val_split   = 0.15
in_channels = 4
patience=10,           # val Dice 连续 10 轮不升则 early stop
lr_factor=0.5,         # 每次降 LR 到原来的 50%
lr_patience=5   
# —————————————————————————————————————————————————————————————————————————————

def main():
    os.environ["nnUNet_raw_data_base"] = nnunet_raw

    segmenter = BaselineUNetSegmenter(
        save_dir=save_dir,
        lr=lr,
        val_split=val_split,
        in_channels=in_channels
    )
    segmenter.fit_from_nnunet(
        nnunet_raw=nnunet_raw,
        task_id=task_id,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers
    )

    results = segmenter.evaluate_from_nnunet(
        nnunet_raw=nnunet_raw,
        task_id=task_id,
        batch_size=batch_size,
        num_workers=num_workers
    )

if __name__ == "__main__":
    main()
