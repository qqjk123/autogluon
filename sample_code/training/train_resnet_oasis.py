import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob

# 1. Configuration and Hyperparameters
# =======================================
# --- Hyperparameter Suggestions ---
# LEARNING_RATE: 0.001 is a good default for Adam. Others to try: 1e-4, 5e-4.
# BATCH_SIZE: 16 or 32 is standard. Larger sizes require more GPU memory.
# NUM_EPOCHS: Set high (e.g., 100) and let Early Stopping find the best epoch.
# PATIENCE: How many epochs to wait for improvement. 5-10 is a common range.
# NUM_SLICES_PER_SCAN: More slices can provide more info but increases training time.
# ----------------------------------
DATA_DIR = './'
CSV_FILE = os.path.join(DATA_DIR, 'oasis_cross-sectional.csv')
IMAGE_DIR = DATA_DIR

NUM_CLASSES = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50  # Set high, Early Stopping will handle the rest
PATIENCE = 10
IMG_SIZE = (208, 176)
NUM_SLICES_PER_SCAN = 10

# 2. Helper Classes and Functions
# =======================================

# ✨ NEW: EarlyStopping class to prevent overfitting
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class RepeatChannel(object):
    def __call__(self, x):
        return x.repeat(3, 1, 1)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.zeros(0), torch.zeros(0)
    return torch.utils.data.dataloader.default_collate(batch)

class OasisDataset(Dataset):
    # ... (OasisDataset class is unchanged) ...
    def __init__(self, df, image_dir, transform=None, num_slices=10):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.num_slices = num_slices
    def __len__(self):
        return len(self.df) * self.num_slices
    def __getitem__(self, idx):
        df_idx = idx // self.num_slices
        slice_idx_in_scan = idx % self.num_slices
        row = self.df.iloc[df_idx]
        session_id = row['ID']
        label = row['CDR_binary']
        base_path = os.path.join(self.image_dir, session_id, 'PROCESSED', 'MPRAGE', 'T88_111')
        img_path_pattern = os.path.join(base_path, f"OAS1_{session_id.split('_')[1]}_{session_id.split('_')[2]}_mpr_*_anon_111_t88_masked_gfc.img")
        img_paths = glob.glob(img_path_pattern)
        if not img_paths: return None
        img_path = img_paths[0]
        try:
            img = nib.load(img_path)
            img_data = img.get_fdata()
            if img_data.ndim == 4:
                img_data = img_data[:, :, :, 0]
            start_slice = (img_data.shape[2] - self.num_slices) // 2
            slice_data = img_data[:, :, start_slice + slice_idx_in_scan]
            if np.max(slice_data) > np.min(slice_data):
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
            image = torch.from_numpy(slice_data).float().unsqueeze(0)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading or processing {img_path}: {e}")
            return None

def load_data(csv_path, test_size=0.2, val_size=0.2):
    # ... (load_data function is unchanged) ...
    df = pd.read_csv(csv_path)
    df['CDR'].fillna(0, inplace=True)
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
    print(f"Training subjects: {len(train_df)} | Validation subjects: {len(val_df)} | Testing subjects: {len(test_df)}")
    return train_df, val_df, test_df

def get_model(num_classes):
    # ... (get_model function is unchanged) ...
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ✨ MODIFIED: train_model function now uses EarlyStopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=7):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    model.to(device)
    
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            dataloader = train_loader if phase == 'train' else val_loader
            model.train() if phase == 'train' else model.eval()

            running_loss, running_corrects = 0.0, 0.0
            dataset_size = len(dataloader.dataset)

            if dataset_size == 0:
                # Handle case where a loader is empty
                print(f"Skipping {phase} phase, dataset is empty.")
                history[f'{phase}_loss'].append(0)
                history[f'{phase}_acc'].append(0)
                continue

            for inputs, labels in dataloader:
                if inputs.size(0) == 0: continue
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0
            epoch_acc = running_corrects / dataset_size if dataset_size > 0 else 0
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Early stopping check (only after validation phase)
            if phase == 'val':
                early_stopping(epoch_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    print("Loading best model weights from checkpoint.")
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    return model, history

# 4. Main Execution Block
# =======================================
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        RepeatChannel(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df, val_df, test_df = load_data(CSV_FILE)
    train_dataset = OasisDataset(train_df, IMAGE_DIR, transform=transform, num_slices=NUM_SLICES_PER_SCAN)
    val_dataset = OasisDataset(val_df, IMAGE_DIR, transform=transform, num_slices=NUM_SLICES_PER_SCAN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model = get_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if len(train_loader.dataset) == 0 and len(val_loader.dataset) == 0:
        print("All data loaders are empty. Exiting.")
    else:
        # Pass patience for early stopping here
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, patience=PATIENCE)
        
        # Save the best model from the checkpoint
        torch.save(model.state_dict(), 'resnet18_oasis_best_model.pth')
        print("\nBest model saved to resnet18_oasis_best_model.pth")

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.axvline(np.argmin(history['val_loss']), linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("Training curve plot saved to training_curves.png")
