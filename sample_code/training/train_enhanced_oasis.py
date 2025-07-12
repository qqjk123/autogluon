import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split

# 导入 MONAI 相关库
from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    ToTensord,
    Resized,
    RandAffined,
    RandFlipd,
)
from monai.networks.nets import ResNet, EfficientNetBN
from monai.metrics import ROCAUCMetric
from monai.networks.nets.resnet import get_inplanes

# ======================================================================================
# 1. 配置区域 (在此处修改参数和模型)
# ======================================================================================
class Config:
    # --- 数据和路径配置 ---
    DATA_DIR = './'
    CSV_FILE = os.path.join(DATA_DIR, 'oasis_cross-sectional.csv')
    IMAGE_DIR = DATA_DIR

    # --- 模型选择 ---
    MODEL_NAME = 'ResNet50'
    
    # --- 训练超参数 ---
    NUM_CLASSES = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    PATIENCE = 10
    
    # --- 3D图像处理配置 ---
    TARGET_SPATIAL_SIZE = (96, 96, 96) 
    
    # --- 硬件与效率配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 10
    USE_AMP = True

    # --- 模型保存 ---
    MODEL_SAVE_PATH = f'enhanced_{MODEL_NAME}_oasis_best_model.pth'

# ======================================================================================
# 2. 增强的早停类 (无变化)
# ======================================================================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode
        if self.mode not in ['min', 'max']:
            raise ValueError("mode 必须是 'min' 或 'max'.")
        self.val_score_min_or_max = np.inf if mode == 'min' else -np.inf

    def __call__(self, current_score, model):
        # Handle nan scores
        if np.isnan(current_score):
            # If score is nan, we can't improve, so we can count it as a non-improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter (score is nan): {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score_to_check = current_score
        if self.best_score is None:
            self.best_score = score_to_check
            self.save_checkpoint(current_score, model)
        elif (self.mode == 'max' and score_to_check < self.best_score + self.delta) or \
             (self.mode == 'min' and score_to_check > self.best_score - self.delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score_to_check
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, current_score, model):
        if self.verbose:
            print(f'Validation score improved ({self.val_score_min_or_max:.6f} --> {current_score:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_score_min_or_max = current_score

# ======================================================================================
# 3. 数据集与数据加载 (代码已更新)
# ======================================================================================
class OasisDataset3D(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        session_id = row['ID']
        label = torch.tensor(row['CDR_binary'], dtype=torch.long)
        base_path = os.path.join(self.image_dir, session_id, 'PROCESSED', 'MPRAGE', 'T88_111')
        img_path_pattern = os.path.join(base_path, f"OAS1_{session_id.split('_')[1]}_{session_id.split('_')[2]}_mpr_*_anon_111_t88_masked_gfc.img")
        img_paths = glob.glob(img_path_pattern)

        if not img_paths: return None
        
        img_path = img_paths[0]
        try:
            img = nib.load(img_path)
            img_data = img.get_fdata(dtype=np.float32)
            if img_data.ndim == 4: img_data = img_data[:, :, :, 0]
            img_data = np.expand_dims(img_data, axis=0) 
            data = {'image': img_data, 'label': label}
            if self.transform: data = self.transform(data)
            return data
        except Exception as e:
            print(f"Error loading or processing {img_path}: {e}")
            return None

def load_data_splits(csv_path, test_size=0.2, val_size=0.2):
    df = pd.read_csv(csv_path)
    # FIX: Updated pandas syntax to avoid warning
    df['CDR'] = df['CDR'].fillna(0)
    df['CDR_binary'] = df['CDR'].apply(lambda x: 0 if x == 0 else 1)
    df = df[df['CDR'] <= 2]
    
    subjects = df['ID'].apply(lambda x: x.split('_')[1]).unique()
    if len(subjects) < 3:
        raise ValueError("Not enough subjects for train/val/test split.")
        
    train_subjects, test_subjects = train_test_split(subjects, test_size=test_size, random_state=42)
    val_split_size = val_size / (1 - test_size)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=val_split_size, random_state=42)
    
    train_df = df[df['ID'].apply(lambda x: x.split('_')[1]).isin(train_subjects)].reset_index(drop=True)
    val_df = df[df['ID'].apply(lambda x: x.split('_')[1]).isin(val_subjects)].reset_index(drop=True)
    test_df = df[df['ID'].apply(lambda x: x.split('_')[1]).isin(test_subjects)].reset_index(drop=True)
    
    print(f"Total Subjects: {len(subjects)}")
    print(f"Training scans: {len(train_df)} | Validation scans: {len(val_df)} | Testing scans: {len(test_df)}")
    return train_df, val_df, test_df

# ======================================================================================
# 4. 模型定义 (无变化)
# ======================================================================================
def get_model(config):
    model_name, num_classes, in_channels = config.MODEL_NAME, config.NUM_CLASSES, 1
    print(f"Initializing 3D model: {model_name}...")
    if model_name == 'ResNet50':
        model = ResNet(block='bottleneck', layers=[3, 4, 6, 3], block_inplanes=get_inplanes(),
                       n_input_channels=in_channels, num_classes=num_classes, spatial_dims=3)
    elif model_name.startswith('EfficientNet'):
        model = EfficientNetBN(model_name=model_name.lower(), pretrained=False,
                               spatial_dims=3, in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model.to(config.DEVICE)

# ======================================================================================
# 5. 训练与评估模块 (代码已更新)
# ======================================================================================
def calculate_metrics(outputs, labels, num_classes):
    """Helper function to calculate accuracy and AUC."""
    # Accuracy
    preds_class = torch.argmax(outputs, dim=1)
    correct_counts = (preds_class == labels).sum().item()
    accuracy = correct_counts / len(labels)
    
    # AUC
    auc_metric = ROCAUCMetric()
    probs = torch.softmax(outputs, dim=1)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    try:
        auc_metric(y_pred=probs, y=labels_onehot)
        auc = auc_metric.aggregate().item()
    except Exception:
        # This happens if a batch has only one class
        auc = np.nan
        
    return accuracy, auc

def train_model(config, model, train_loader, val_loader):
    device = config.DEVICE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    # FIX: Updated GradScaler syntax
    scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)
    
    early_stopping = EarlyStopping(patience=config.PATIENCE, verbose=True, path=config.MODEL_SAVE_PATH, mode='max')
    
    history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'val_acc': [], 'val_auc': [], 'val_score': []}
    
    print(f"\n--- Starting Training on {device} ---")
    for epoch in range(config.NUM_EPOCHS):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        # ## NEW: Lists to store train outputs for metric calculation
        all_train_outputs, all_train_labels = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        for batch_data in train_pbar:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            # FIX: Updated autocast syntax
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            # ## NEW: Store outputs for metric calculation after epoch
            all_train_outputs.append(outputs.detach().cpu())
            all_train_labels.append(labels.detach().cpu())
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)
        
        # ## NEW: Calculate and display training metrics ##
        all_train_outputs = torch.cat(all_train_outputs)
        all_train_labels = torch.cat(all_train_labels)
        train_acc, train_auc = calculate_metrics(all_train_outputs, all_train_labels, config.NUM_CLASSES)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        print(f"Epoch {epoch+1} Train -> Loss: {epoch_train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")

        # --- 验证阶段 ---
        model.eval()
        all_val_outputs, all_val_labels = [], []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Validate]")
            for batch_data in val_pbar:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.USE_AMP):
                    outputs = model(inputs)
                all_val_outputs.append(outputs.cpu())
                all_val_labels.append(labels.cpu())

        all_val_outputs = torch.cat(all_val_outputs)
        all_val_labels = torch.cat(all_val_labels)
        val_acc, val_auc = calculate_metrics(all_val_outputs, all_val_labels, config.NUM_CLASSES)
        
        # ## NEW: Handle nan AUC for combined score ##
        if np.isnan(val_auc):
            combined_score = val_acc # Fallback to accuracy if AUC is not available
        else:
            combined_score = val_acc + val_auc
        
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_score'].append(combined_score)
        
        print(f"Epoch {epoch+1} Validate -> Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, Score: {combined_score:.4f}")
        
        early_stopping(combined_score, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
        scheduler.step()

    print("\n--- Training Finished ---")
    print(f"Best validation score: {early_stopping.best_score:.4f}")
    
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    print(f"Best model loaded from {config.MODEL_SAVE_PATH}")
    return model, history

# ## NEW: Function to evaluate on the test set ##
@torch.no_grad()
def evaluate_on_test(config, model, test_loader):
    print("\n--- Starting Test Set Evaluation ---")
    model.to(config.DEVICE)
    model.eval()
    
    all_test_outputs, all_test_labels = [], []
    test_pbar = tqdm(test_loader, desc="[Testing]")
    for batch_data in test_pbar:
        inputs = batch_data["image"].to(config.DEVICE)
        labels = batch_data["label"].to(config.DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.USE_AMP):
            outputs = model(inputs)
        all_test_outputs.append(outputs.cpu())
        all_test_labels.append(labels.cpu())

    all_test_outputs = torch.cat(all_test_outputs)
    all_test_labels = torch.cat(all_test_labels)
    test_acc, test_auc = calculate_metrics(all_test_outputs, all_test_labels, config.NUM_CLASSES)
    
    print("\n" + "="*40)
    print("✅ Test Set Results:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   AUC:      {test_auc:.4f}")
    print("="*40)

# ======================================================================================
# 6. 主执行函数 (代码已更新)
# ======================================================================================
def main():
    config = Config()
    print("="*60)
    print(f"Configuration: Model={config.MODEL_NAME}, Device={config.DEVICE}")
    print(f"Hyperparameters: Epochs={config.NUM_EPOCHS}, Batch Size={config.BATCH_SIZE}, LR={config.LEARNING_RATE}")
    print("="*60 + "\n")

    train_transforms = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=config.TARGET_SPATIAL_SIZE, mode="trilinear", align_corners=False),
        RandFlipd(keys="image", prob=0.5, spatial_axis=[0, 1, 2]),
        RandAffined(keys='image', mode='bilinear', prob=0.5, rotate_range=(np.pi/12, np.pi/12, np.pi/12), scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"], track_meta=False),
    ])

    val_test_transforms = Compose([
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=config.TARGET_SPATIAL_SIZE, mode="trilinear", align_corners=False),
        ToTensord(keys=["image", "label"], track_meta=False),
    ])

    # ## NEW: Get all data splits including test ##
    train_df, val_df, test_df = load_data_splits(config.CSV_FILE)

    def collate_fn_filter_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return list_data_collate(batch) if batch else {'image': torch.zeros(0), 'label': torch.zeros(0)}

    train_dataset = OasisDataset3D(train_df, config.IMAGE_DIR, transform=train_transforms)
    val_dataset = OasisDataset3D(val_df, config.IMAGE_DIR, transform=val_test_transforms)
    # ## NEW: Create test dataset ##
    test_dataset = OasisDataset3D(test_df, config.IMAGE_DIR, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn_filter_none, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn_filter_none, pin_memory=True)
    # ## NEW: Create test loader ##
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn_filter_none, pin_memory=True)
    
    if len(train_loader.dataset) == 0:
        print("Training dataset is empty. Exiting.")
        return

    model = get_model(config)
    model, history = train_model(config, model, train_loader, val_loader)

    # ## NEW: Run evaluation on the test set after training ##
    if len(test_loader.dataset) > 0:
        evaluate_on_test(config, model, test_loader)
    else:
        print("Test dataset is empty. Skipping test evaluation.")

    print(f"\n✅ All tasks completed. Best model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
