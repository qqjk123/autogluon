from autogluon.multimodal.learners import BaselineUNETRSegmenter

seg = BaselineUNETRSegmenter(save_dir="checkpoints_unetr/001")
# 训练
seg.fit_from_nnunet(
    "/workspace/transfer/nnUnet_raw",
    "002_BraTS2023_MEN",
    #"001_BraTS2023_SEG",
    epochs=50,
    batch_size=4,
    num_workers=16,
    cache_rate=0.5
)

# 评估
metrics = seg.evaluate_from_nnunet(
    "/workspace/transfer/nnUnet_raw",
    "002_BraTS2023_MEN",
    #"001_BraTS2023_SEG",
    batch_size=4,       # 指定 batch_size
    num_workers=16,     # 指定 num_workers
    cache_rate=0.5      # 指定 cache_rate
)
print(metrics)
