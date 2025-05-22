# Dice Enhanced
from autogluon.multimodal import MultiModalPredictor

hyperparameters = {
    "optimization": {
        # 混合精度训练，节省显存、加速
        "fp16": True,              
        # 每张卡 batch size，4090 24 GB 可以试 8 或 16，根据图像大小调小／调大
        "per_gpu_batch_size": 2,   
        # 小数据集适当增加epoch次数
        "max_epochs": 10,          
        # 学习率微调：小数据集、少样本，用小一点的 LR 更稳定
        "lr": 1e-4,                
        # L2 正则，防止过拟合
        "weight_decay": 1e-4,      
        # 梯度累积：如果想试更大等效 batch，再叠几步
        "gradient_accumulation_steps": 2,
        # 早停配置，val 不提升就停
        "early_stop": {"monitor": "val_dice", "patience": 3},
        # 数据加载预取
        "num_workers": 64,
        "loss_function": "dice"
    }
}

predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    label="label",
    path="automm/automm_dataset002",
    presets="medium_quality",   
    eval_metric="dice",              # 训练/调参时的评估指标
    validation_metric="dice",      # 可选 medium_quality / high_quality
    hyperparameters=hyperparameters,
)

print("Validation metric is:", predictor._learner._validation_metric_name)

predictor.fit(
    train_data="/workspace/autogluon_original_Dataset002_BraTS2023_SEG/train.csv",
    tuning_data="/workspace/autogluon_original_Dataset002_BraTS2023_SEG/val.csv",
    time_limit=18000,                 # 5 小时
)
