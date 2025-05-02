import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from monai.networks.nets import resnet18
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class ResNetClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.resnet = resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
            pretrained=False,
        )

    def forward(self, x):
        return self.resnet(x)

class ResNetPredictor:
    def __init__(self, label: str, path: str, batch_size=16):
        self.label = label
        self.path = path
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(path, exist_ok=True)
        self.in_channels = 1
        self.num_classes = 3  # default

    def _preprocess_dataset(self, dataset):
        images, labels = dataset

        if isinstance(images, torch.Tensor):
            images = images.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        images = np.transpose(images, (1, 0, 2, 3, 4))  # (C, N, D, H, W) -> (N, C, D, H, W)
        labels = np.squeeze(labels, axis=-1)  # (N,)

        X_min, X_max = images.min(), images.max()
        images = (images - X_min) / (X_max - X_min + 1e-8)

        X_tensor = torch.tensor(images, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor)

    def fit(self, train_data, val_data=None, time_limit=None, **kwargs):
        train_dataset = self._preprocess_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if val_data:
            val_dataset = self._preprocess_dataset(val_data)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.num_classes = len(torch.unique(train_dataset.tensors[1]))
        self.model = ResNetClassifier(in_channels=self.in_channels, num_classes=self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        import time
        start_time = time.time()
        max_epochs = 100

        best_combined_score = -np.inf
        best_model_state = None

        for epoch in range(max_epochs):
            epoch_loss = 0
            self.model.train()

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if time_limit and (time.time() - start_time > time_limit):
                print(f"Stopping early at epoch {epoch+1} due to time limit.")
                break

            if val_data:
                self.model.eval()
                all_preds = []
                all_labels = []
                all_probs = []

                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                        val_out = self.model(X_val_batch)
                        val_probs_batch = torch.softmax(val_out, dim=1)
                        val_preds_batch = val_probs_batch.argmax(dim=1)

                        all_preds.append(val_preds_batch.cpu().numpy())
                        all_labels.append(y_val_batch.cpu().numpy())
                        all_probs.append(val_probs_batch.cpu().numpy())

                preds = np.concatenate(all_preds)
                labels = np.concatenate(all_labels)
                probs = np.concatenate(all_probs)

                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average="macro")
                if self.num_classes == 2:
                    val_auc = roc_auc_score(labels, probs[:, 1])
                else:
                    val_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

                combined_score = (acc + val_auc) / 2

                if (epoch + 1) % 10 == 0:
                    print(f"[Epoch {epoch+1}] Val Accuracy: {acc*100:.2f}%, Val F1: {f1*100:.2f}%, Val AUC: {val_auc:.4f}")

                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_model_state = self.model.state_dict()

        # Save the best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            torch.save(self.model.state_dict(), os.path.join(self.path, "best_model.pt"))
            print(f"Best model saved with combined score: {best_combined_score:.4f}")

        # Save the final model too
        torch.save(self.model.state_dict(), os.path.join(self.path, "model.pt"))

    def evaluate(self, test_data, metrics=["accuracy"]):
        test_dataset = self._preprocess_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)

        results = {}

        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(labels, preds)
        if "f1" in metrics:
            results["f1"] = f1_score(labels, preds, average="macro")
        if "auc" in metrics:
            if self.num_classes == 2:
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            results["auc"] = auc

        return results

    def predict(self, input_dict):
        self.model.eval()
        imgs = input_dict["input"]

        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        tensor_list = []
        for img in imgs:
            if img.ndim == 5:  # (C, D, H, W)
                img = np.transpose(img, (1, 0, 2, 3, 4))  # (N, C, D, H, W)
            if img.ndim == 4:
                img = np.expand_dims(img, axis=0)  # batch dim
            tensor_list.append(img.astype(np.float32))

        tensor = torch.tensor(np.vstack(tensor_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            pred = output.argmax(dim=1).cpu().numpy()
        return pred
