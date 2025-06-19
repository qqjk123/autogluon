#v4
# v3 
# Resnet - 最终版 (支持 ResNet50/101, EfficientNet, Swin Transformer)
# [修改] 已整合方案一: 使用 Accuracy + AUC 的综合得分来指导验证
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# 导入 MedMNIST 和 MONAI 相关库
import medmnist
from medmnist import INFO

from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    ToTensord,
    Resized,
    RandRotate90d,
    RandAffined,
    RandFlipd,
)
# [新增] 导入 SwinTransformer, 此处需要从 monai.networks.nets 导入
from monai.networks.nets import ResNet, EfficientNetBN
from monai.metrics import ROCAUCMetric
from monai.networks.nets.resnet import get_inplanes

# ======================================================================================
# 1. 配置区域 (可在此切换模型)
# ======================================================================================
# 'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d'
class Config:
    # --- 实验与数据配置 ---
    DATA_FLAG = 'nodulemnist3d'
    
    # --- 模型选择 (在此处修改以切换模型) ---
    MODEL_NAME = 'ResNet50'          # 选项1: 原始基线模型
    #MODEL_NAME = 'ResNet101'         # 选项2: 更深的ResNet
    #MODEL_NAME = 'EfficientNet-B4'   # 选项3: 高效的CNN架构 (推荐)
    #MODEL_NAME = 'SwinTransformer'   # 选项4: 先进的Transformer架构 (推荐)
    
    EXPERIMENT_NAME = f"{MODEL_NAME}_{DATA_FLAG}_Final_CombinedMetric" # 为本次实验命名
    
    # --- 训练超参数 ---
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # --- 硬件与效率配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 16
    USE_AMP = True
    
    # --- 模型保存 ---
    MODEL_SAVE_DIR = "./saved_models"

# ======================================================================================
# 2. 辅助函数 (数据加载和模型定义) - 无需修改
# ======================================================================================
class MedMNIST3DWrapper(Dataset):
    def __init__(self, medmnist_dataset, transform):
        self.medmnist_dataset = medmnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.medmnist_dataset)

    def __getitem__(self, idx):
        image, label = self.medmnist_dataset[idx]
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        data = {'image': image, 'label': label}
        if self.transform:
            data = self.transform(data)
        return data

def get_data_loaders(config):
    print(f"准备数据集: {config.DATA_FLAG}...")
    info = INFO[config.DATA_FLAG]
    DataClass = getattr(medmnist.dataset, f"{info['python_class']}")
    
    target_spatial_size = (32, 32, 32)

    train_transforms = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=target_spatial_size, mode="trilinear", align_corners=False),
        RandFlipd(keys="image", spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90d(keys="image", prob=0.5, max_k=3),
        RandAffined(
            keys='image', mode='bilinear', prob=0.8, spatial_size=target_spatial_size,
            rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
            scale_range=(0.2, 0.2, 0.2), padding_mode='border'
        ),
        ToTensord(keys=["image", "label"], track_meta=False),
    ])

    val_test_transforms = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=target_spatial_size, mode="trilinear", align_corners=False),
        ToTensord(keys=["image", "label"], track_meta=False),
    ])

    train_dataset_raw = DataClass(split='train', download=True)
    train_ds = MedMNIST3DWrapper(train_dataset_raw, train_transforms)
    
    val_dataset_raw = DataClass(split='val', download=True)
    val_ds = MedMNIST3DWrapper(val_dataset_raw, val_test_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  
        num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,  
        num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True
    )
    
    print(f"数据加载完成. 任务: {info['task']}, 类别数: {len(info['label'])}")
    return train_loader, val_loader, info, target_spatial_size

def get_model(config, info, img_size):
    model_name, num_classes = config.MODEL_NAME, len(info['label'])
    print(f"初始化模型: {model_name}...")
    in_channels = 1 
    
    if model_name == 'ResNet50':
        model = ResNet(block='bottleneck', layers=[3, 4, 6, 3], block_inplanes=get_inplanes(), 
                       n_input_channels=in_channels, num_classes=num_classes, spatial_dims=3)
    
    elif model_name == 'ResNet101':
        model = ResNet(block='bottleneck', layers=[3, 4, 23, 3], block_inplanes=get_inplanes(), 
                       n_input_channels=in_channels, num_classes=num_classes, spatial_dims=3)

    elif model_name.startswith('EfficientNet'):
        model = EfficientNetBN(model_name=model_name.lower(), pretrained=False, 
                               spatial_dims=3, in_channels=in_channels, num_classes=num_classes)
        
    elif model_name == 'SwinTransformer':
        model = SwinTransformer(
            in_chans=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=(4, 4, 4),
            window_size=(4, 4, 4), 
            depths=[2, 2, 6, 2],    # Swin-T (Tiny) 配置
            num_heads=[3, 6, 12, 24],
            use_v2=False, # 使用 V1 版本的 Swin
        )

    else:
        raise ValueError(f"模型 '{model_name}' 不被支持.")
        
    return model.to(config.DEVICE)


# ======================================================================================
# 3. 训练与评估模块 (已按方案一修改)
# ======================================================================================
def run_training(config, model, train_loader, val_loader, info):
    task = info['task']
    device = config.DEVICE

    loss_function = nn.CrossEntropyLoss() if task != 'multi-label, binary-class' else nn.BCEWithLogitsLoss()
    print(f"任务为 {task}, 使用 {loss_function.__class__.__name__}. 主要评估指标: Combined Score (AUC + Accuracy).")
        
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)
    
    # best_metric 现在将存储最佳的 "综合得分"
    best_metric, best_metric_epoch = -1, -1
    
    print(f"\n--- 开始训练: {config.EXPERIMENT_NAME} ---")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [训练]", unit="batch")
        for batch_data in progress_bar:
            inputs, labels = batch_data["image"].to(device, non_blocking=True), batch_data["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                outputs = model(inputs)
                if isinstance(loss_function, nn.CrossEntropyLoss):
                    labels = labels.squeeze(1).long()
                else:  
                    labels = labels.float()
                loss = loss_function(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
        
        scheduler.step()
        print(f"Epoch {epoch + 1} 平均训练损失: {epoch_loss / len(train_loader):.4f}")

        # --- 验证逻辑 ---
        model.eval()
        auc_metric = ROCAUCMetric()
        all_val_outputs, all_val_labels = [], []
        with torch.no_grad():
            for val_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [验证]", unit="batch"):
                val_images = val_data["image"].to(device, non_blocking=True)
                val_labels = val_data["label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                    val_outputs = model(val_images)
                all_val_outputs.append(val_outputs)
                all_val_labels.append(val_labels)

        all_val_outputs = torch.cat(all_val_outputs, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)

        # 计算 Accuracy 和 AUC
        if task == 'multi-label, binary-class':
            val_probs = torch.sigmoid(all_val_outputs)
            auc_metric(y_pred=val_probs, y=all_val_labels.long())
            preds_class = val_probs > 0.5
            acc_result = (preds_class == all_val_labels.bool()).sum().item() / all_val_labels.numel()
        else:
            val_probs = torch.softmax(all_val_outputs, dim=1)
            val_labels_1d = all_val_labels.squeeze(1).long()
            num_classes = val_probs.shape[1]
            val_labels_onehot = F.one_hot(val_labels_1d, num_classes=num_classes).float()
            y_pred_classes = torch.argmax(all_val_outputs, dim=1)
            correct = (y_pred_classes == val_labels_1d).sum().item()
            acc_result = correct / val_labels_1d.numel()
            auc_metric(y_pred=val_probs, y=val_labels_onehot)
            
        auc_result = auc_metric.aggregate().item()

        # ======================= [核心修改] =======================
        # 1. 计算综合得分 (简单相加)
        combined_score = acc_result + auc_result

        print(f"Epoch {epoch + 1} 验证 -> AUC: {auc_result:.4f}, Accuracy: {acc_result:.4f}, Combined Score: {combined_score:.4f}")

        # 2. 使用 "Combined Score" 来判断和保存最佳模型
        if combined_score > best_metric:
            best_metric, best_metric_epoch = combined_score, epoch + 1
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"best_{config.EXPERIMENT_NAME}.pth")
            torch.save(model.state_dict(), save_path)
            # 3. 更新提示信息
            print(f"🎉 新的最佳模型! (基于 Combined Score) 已保存到: {save_path}")
        # ==========================================================

    print(f"--- 训练完成 ---")
    print(f"最佳验证 Combined Score: {best_metric:.4f} (在 Epoch {best_metric_epoch})")

# ======================================================================================
# 4. Test 集评估 (无需修改)
# ======================================================================================
@torch.no_grad()
def evaluate_on_test(config, info, img_size):
    print("\n========== 开始 Test 评估 ==========")

    test_transforms = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=img_size, mode="trilinear", align_corners=False),
        ToTensord(keys=["image", "label"], track_meta=False),
    ])

    DataClass = getattr(medmnist.dataset, f"{info['python_class']}")
    test_dataset_raw = DataClass(split="test", download=True)
    test_ds = MedMNIST3DWrapper(test_dataset_raw, test_transforms)

    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        #num_workers=config.NUM_WORKERS, collate_fn=list_data_collate, pin_memory=True
        num_workers=0, collate_fn=list_data_collate, pin_memory=True
    )

    model = get_model(config, info, img_size)
    best_ckpt_path = os.path.join(config.MODEL_SAVE_DIR, f"best_{config.EXPERIMENT_NAME}.pth")
    
    if not os.path.exists(best_ckpt_path):
        print(f"错误：找不到训练好的模型权重 '{best_ckpt_path}'。请先完成训练。")
        return

    model.load_state_dict(torch.load(best_ckpt_path, map_location=config.DEVICE))
    model.eval()

    all_outputs, all_labels = [], []
    for batch in tqdm(test_loader, desc="Test 推断", unit="batch"):
        imgs = batch["image"].to(config.DEVICE, non_blocking=True)
        labels = batch["label"].to(config.DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            logits = model(imgs)
        all_outputs.append(logits)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 针对多分类任务计算指标
    if info['task'] != 'multi-label, binary-class':
        labels_1d = all_labels.squeeze(1).long()
        preds_cls = torch.argmax(all_outputs, dim=1)
        accuracy = (preds_cls == labels_1d).float().mean().item()

        probs = torch.softmax(all_outputs, dim=1)
        one_hot = F.one_hot(labels_1d, num_classes=probs.shape[1]).float()
        auc_metric = ROCAUCMetric()
        auc_metric(y_pred=probs, y=one_hot)
        auc = auc_metric.aggregate().item()
    # 针对多标签任务计算指标
    else:
        probs = torch.sigmoid(all_outputs)
        preds_cls = probs > 0.5
        accuracy = (preds_cls == all_labels.bool()).float().mean().item()

        auc_metric = ROCAUCMetric()
        auc_metric(y_pred=probs, y=all_labels.long())
        auc = auc_metric.aggregate().item()

    print(f"\n✅ Test 结果 ({config.EXPERIMENT_NAME}) →  ACC: {accuracy:.4f} | AUC: {auc:.4f}")
    print("=====================================")


# ======================================================================================
# 5. 主执行函数 (无需修改)
# ======================================================================================
def main():
    config = Config()
    print("="*60)
    print(f"配置: Dataset={config.DATA_FLAG}, Model={config.MODEL_NAME}, Device={config.DEVICE}")
    print(f"实验名称: {config.EXPERIMENT_NAME}")
    print(f"超参数: Epochs={config.NUM_EPOCHS}, Batch Size={config.BATCH_SIZE}, LR={config.LEARNING_RATE}, AMP={'启用' if config.USE_AMP else '禁用'}")
    print("="*60 + "\n")

    train_loader, val_loader, info, img_size = get_data_loaders(config)
    model = get_model(config, info, img_size)
    
    run_training(config, model, train_loader, val_loader, info)
    
    # 训练结束后，自动在测试集上评估当前配置的最佳模型
    evaluate_on_test(config, info, img_size)
    
    print("\n所有任务完成!")

if __name__ == "__main__":
    main()
