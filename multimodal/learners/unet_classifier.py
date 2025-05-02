import os
import uuid
import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import UNet
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class UNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=16,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=2,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        features = self.unet(x)
        pooled = self.pool(features).view(features.size(0), -1)
        return self.fc(pooled)


class UNetPredictor:
    def __init__(self, label: str, path: str):
        self.label = label
        self.path = path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(path, exist_ok=True)
        self.in_channels = 1
        self.num_classes = 3  # default

    def _preprocess_dataset(self, dataset):
        """
        Accepts (images, labels):
        - images: (C, N, D, H, W)
        - labels: (N, 1)
        """
        images, labels = dataset

        if isinstance(images, torch.Tensor):
            images = images.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # images: (C, N, D, H, W) -> (N, C, D, H, W)
        images = np.transpose(images, (1, 0, 2, 3, 4))

        labels = np.squeeze(labels, axis=-1)  # (N,)

        X_min, X_max = images.min(), images.max()
        images = (images - X_min) / (X_max - X_min + 1e-8)

        X_tensor = torch.tensor(images, dtype=torch.float32)  # already (N, C, D, H, W)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        return X_tensor.to(self.device), y_tensor.to(self.device)

    def fit(self, train_data, val_data=None, time_limit=None, **kwargs):
        X, y = self._preprocess_dataset(train_data)
        if val_data:
            X_val, y_val = self._preprocess_dataset(val_data)

        self.num_classes = len(torch.unique(y))
        self.model = UNetClassifier(in_channels=self.in_channels, num_classes=self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        import time
        start_time = time.time()
        max_epochs = 100

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            out = self.model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if time_limit and (time.time() - start_time > time_limit):
                print(f"Stopping early at epoch {epoch+1} due to time limit.")
                break

            if val_data and (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val)
                    val_pred = val_out.argmax(dim=1)
                    acc = (val_pred == y_val).float().mean().item() * 100
                    f1 = f1_score(y_val.cpu().numpy(), val_pred.cpu().numpy(), average="macro") * 100
                    print(f"[Epoch {epoch+1}] Validation Accuracy: {acc:.2f}%, Validation F1: {f1:.2f}%")
                self.model.train()

        torch.save(self.model.state_dict(), os.path.join(self.path, "model.pt"))

    def evaluate(self, test_data, metrics=["accuracy"]):
        X, y = self._preprocess_dataset(test_data)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).argmax(dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        results = {}
        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(y_true, preds)
        if "f1" in metrics:
            results["f1"] = f1_score(y_true, preds, average="macro")  # macro适合多分类
        return results

    def predict(self, input_dict):
        self.model.eval()
        imgs = input_dict["input"]

        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        tensor_list = []
        for img in imgs:
            if img.ndim == 5:  # (C, D, H, W)
                img = np.transpose(img, (1, 0, 2, 3, 4))  # make (N, C, D, H, W)
            if img.ndim == 4:
                img = np.expand_dims(img, axis=0)  # Add batch dim
            tensor_list.append(img.astype(np.float32))

        tensor = torch.tensor(np.vstack(tensor_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1).cpu().numpy()
        return pred
