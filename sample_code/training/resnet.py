# Resnet 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

import medmnist
from medmnist import INFO

from monai.data import list_data_collate
# [最终修复] 移除所有自动处理通道的Transform
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    ToTensord,
    ToTensor,
    Resized
)
from monai.networks.nets import DenseNet121, EfficientNetBN, ResNet, ViT
from monai.metrics import ROCAUCMetric, get_confusion_matrix, compute_confusion_matrix_metric
from monai.networks.nets.resnet import get_inplanes
import torch.nn.functional as F

# ======================================================================================
# 1. 配置区域
# ======================================================================================
#'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d'
class Config:
    DATA_FLAG = 'adrenalmnist3d'
    MODEL_NAME = 'ResNet50'
    NUM_EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 16
    MODEL_SAVE_DIR = "./saved_models"

# ======================================================================================
# 2. 辅助函数 (数据加载和模型定义)
# ======================================================================================
class MedMNIST3DWrapper(Dataset):
    def __init__(self, medmnist_dataset, transform):
        self.medmnist_dataset = medmnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.medmnist_dataset)

    def __getitem__(self, idx):
        image, label = self.medmnist_dataset[idx]
        
        # [最终修复] 使用Numpy手动添加通道维度，确保形状正确
        # 原始形状: (28, 28, 28) -> 新形状: (1, 28, 28, 28)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # 将处理好的Numpy数组放入字典，以供后续transform使用
        data = {'image': image, 'label': label}
        
        # 应用剩余的transform（缩放和转为张量）
        if self.transform:
            data = self.transform(data)
            
        return data

def get_data_loaders(config):
    print(f"准备数据集: {config.DATA_FLAG}...")
    info = INFO[config.DATA_FLAG]
    DataClass = getattr(medmnist.dataset, f"{info['python_class']}")
    img_size = (28, 28, 28)
    
    # [最终修复] 简化Transform流程，移除所有通道处理操作
    transforms = Compose([
        ScaleIntensityd(keys="image"),
        Resized(
        keys="image",                 # string or list both fine
        spatial_size=(32, 32, 32),    # make each axis ≥32 so DenseNet works
        mode="trilinear",             # interpolation for images
        align_corners=False           # optional, safe for most cases
        ),
        ToTensord(keys=["image", "label"], track_meta=False),
    ])

    train_dataset_raw = DataClass(split='train', download=True)
    train_ds = MedMNIST3DWrapper(train_dataset_raw, transforms)
    
    val_dataset_raw = DataClass(split='val', download=True)
    val_ds = MedMNIST3DWrapper(val_dataset_raw, transforms)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, collate_fn=list_data_collate
    )
    
    # 注意：您的环境将organmnist3d识别为multi-class，代码将据此调整。
    # 这可能与官方文档不符，或是由于medmnist库版本较旧。
    print(f"数据加载完成. 任务: {info['task']}, 类别数: {len(info['label'])}")
    return train_loader, val_loader, info, img_size

def get_model(config, info, img_size):
    model_name, num_classes = config.MODEL_NAME, len(info['label'])
    print(f"初始化模型: {model_name}...")
    in_channels = 1 # 我们手动保证了输入通道为1
    
    if model_name == 'DenseNet121':
        model = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=num_classes)
    elif model_name == 'EfficientNet-B0':
        model = EfficientNetBN(model_name='efficientnet-b0', spatial_dims=3, in_channels=in_channels, num_classes=num_classes, pretrained=False)
    elif model_name == 'ResNet50':
        model = ResNet(block='bottleneck', layers=[3, 4, 6, 3], block_inplanes=get_inplanes(), spatial_dims=3, n_input_channels=in_channels, num_classes=num_classes)
    elif model_name == 'ViT':
        model = ViT(in_channels=in_channels, img_size=img_size, patch_size=(7, 7, 7), classification=True, num_classes=num_classes)
    else:
        raise ValueError(f"模型 '{model_name}' 不被支持.")
    return model.to(config.DEVICE)


# ======================================================================================
# 3. 训练与评估模块
# ======================================================================================
def run_training(config, model, train_loader, val_loader, info):
    task = info['task']
    device = config.DEVICE

    if task == 'multi-label, binary-class':
        loss_function = nn.BCEWithLogitsLoss()
        primary_metric_name = "AUC"
        print("任务为多标签分类, 使用 BCEWithLogitsLoss. 主要评估指标: AUC.")
    else: # 您的环境会进入此分支
        loss_function = nn.CrossEntropyLoss()
        primary_metric_name = "Accuracy"
        print(f"任务为 {task}, 使用 CrossEntropyLoss. 主要评估指标: Accuracy.")
        
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    best_metric, best_metric_epoch = -1, -1
    
    print(f"\n--- 开始训练模型: {config.MODEL_NAME} on {config.DATA_FLAG} ---")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [训练]", unit="batch")
        for batch_data in progress_bar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(loss_function, nn.CrossEntropyLoss):
                labels = labels.squeeze(1).long()
            else: 
                labels = labels.float()
                
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch + 1} 平均训练损失: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        auc_metric = ROCAUCMetric()
        all_val_outputs, all_val_labels = [], []
        with torch.no_grad():
            for val_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [验证]", unit="batch"):
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                val_outputs = model(val_images)
                all_val_outputs.append(val_outputs)
                all_val_labels.append(val_labels)
        all_val_outputs = torch.cat(all_val_outputs, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)

        if task == 'multi-label, binary-class':
            val_probs = torch.sigmoid(all_val_outputs)
            auc_metric(y_pred=val_probs, y=all_val_labels.long())
            preds_class = val_probs > 0.5
            acc_result = (preds_class == all_val_labels.bool()).sum().item() / all_val_labels.numel()
        else: # 您的环境会进入此分支
            val_probs = torch.softmax(all_val_outputs, dim=1)      # (N, C)
            val_labels_1d = all_val_labels.squeeze(1).long()       # (N,)
            
            # 关键：把 1-D label 转 one-hot，转成 float 供 auc 计算
            num_classes = val_probs.shape[1]
            val_labels_onehot = F.one_hot(val_labels_1d, num_classes=num_classes).float()
            
            # Accuracy 仍用 1-D label 计算
            y_pred_classes = torch.argmax(all_val_outputs, dim=1)
            correct = (y_pred_classes == val_labels_1d).sum().item()
            acc_result = correct / val_labels_1d.numel()
            
            # AUC 用 one-hot label
            auc_metric(y_pred=val_probs, y=val_labels_onehot)
            
        auc_result = auc_metric.aggregate().item()
        print(f"Epoch {epoch + 1} 验证 -> AUC: {auc_result:.4f}, Accuracy: {acc_result:.4f}")

        current_metric = auc_result if primary_metric_name == "AUC" else acc_result
        if current_metric > best_metric:
            best_metric, best_metric_epoch = current_metric, epoch + 1
            os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
            save_path = os.path.join(config.MODEL_SAVE_DIR, f"best_{config.MODEL_NAME}_{config.DATA_FLAG}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🎉 新的最佳模型! (基于 {primary_metric_name}) 已保存到: {save_path}")

    print(f"--- 训练完成 ---")
    print(f"最佳验证 {primary_metric_name}: {best_metric:.4f} (在 Epoch {best_metric_epoch})")

# ======================================================================================
# 4. Test 集评估
# ======================================================================================
@torch.no_grad()
def evaluate_on_test(config, info, img_size):
    print("\n========== 开始 Test 评估 ==========")

    # -------- 1. 组装与训练同样的 transforms --------
    test_transforms = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=(32, 32, 32),
                mode="trilinear", align_corners=False),
        ToTensord(keys=["image", "label"], track_meta=False),
    ])

    # -------- 2. 加载 Test 数据 --------
    DataClass = getattr(medmnist.dataset, f"{info['python_class']}")
    test_dataset_raw = DataClass(split="test", download=True)
    test_ds = MedMNIST3DWrapper(test_dataset_raw, test_transforms)

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=list_data_collate,
    )

    # -------- 3. 构建模型并载入最佳权重 --------
    model = get_model(config, info, img_size)
    best_ckpt = os.path.join(config.MODEL_SAVE_DIR,
                             f"best_{config.MODEL_NAME}_{config.DATA_FLAG}.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=config.DEVICE))
    model.eval()

    # -------- 4. 推断并收集输出 --------
    all_outputs, all_labels = [], []
    for batch in tqdm(test_loader, desc="Test 推断", unit="batch"):
        imgs = batch["image"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)
        logits = model(imgs)
        all_outputs.append(logits)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)         # (N, C)
    all_labels  = torch.cat(all_labels,  dim=0)         # (N, 1)

    # -------- 5. 计算 Accuracy --------
    labels_1d = all_labels.squeeze(1).long()
    preds_cls  = torch.argmax(all_outputs, dim=1)
    accuracy   = (preds_cls == labels_1d).float().mean().item()

    # -------- 6. 计算 Macro-AUC --------
    probs      = torch.softmax(all_outputs, dim=1)
    one_hot    = torch.nn.functional.one_hot(labels_1d,
                                             num_classes=probs.shape[1]).float()
    auc_metric = ROCAUCMetric()
    auc_metric(y_pred=probs, y=one_hot)
    auc        = auc_metric.aggregate().item()

    print(f"\n✅ Test 结果 →  ACC: {accuracy:.4f} | AUC: {auc:.4f}")
    print("=====================================")



# ======================================================================================
# 5. 主执行函数
# ======================================================================================
def main():
    config = Config()
    print("="*50)
    print(f"配置: Dataset={config.DATA_FLAG}, Model={config.MODEL_NAME}, Device={config.DEVICE}")
    print("="*50 + "\n")

    train_loader, val_loader, info, img_size = get_data_loaders(config)
    model = get_model(config, info, img_size)
    run_training(config, model, train_loader, val_loader, info)
    
    print("\n所有任务完成!")

if __name__ == "__main__":
    main()
    evaluate_on_test(Config(), INFO[Config().DATA_FLAG],
                     img_size=(28, 28, 28))
