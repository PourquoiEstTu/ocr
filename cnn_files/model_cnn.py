"""
CNN OCR training script using pixel features.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib

import preprocessing_cnn as pre

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIR = r"/u50/chandd9/al3/ocr/cnn_files/outputs"
FEATURE_DIR = f"/u50/chandd9/al3/ocr-pixel-nested-V2-fixed"

os.makedirs(DIR, exist_ok=True)

dimension = 64  # resized to 64x64 pixels

# CNN MODEL
class CNNModel(nn.Module):
    def __init__(self, input_size=(dimension, dimension), out_size=10):
        super().__init__()

        self.input_h, self.input_w = input_size

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop2d = nn.Dropout2d(0.2)

        # Compute flattened size
        # h = self.input_h // 2 // 2 // 2
        # w = self.input_w // 2 // 2 // 2
        fc_input_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, out_size)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B,1,H,W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = torch.flatten(x, 1)     # flatten
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# MAIN
if __name__ == "__main__":

    # Load features and labels
    X, y, file_names = pre.get_pixels_and_labels(
        FEATURE_DIR,
        f"{FEATURE_DIR}/ordered_labels.npy",
    )

    print(f"Total samples: {len(X)}")

    # Shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, y, file_names = X[idx], np.array(y)[idx], np.array(file_names)[idx]

    # Train-test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]
    test_file_names = file_names[split:]

    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    num_classes = len(le.classes_)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # CNN expects (B,1,H,W)
    # If images are 32×32 pixels:
    X_train = X_train.view(-1, 1, dimension, dimension)
    X_test = X_test.view(-1, 1, dimension, dimension)

    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    # Dataloaders
    batch_size = 16
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # Model
    model = CNNModel(out_size=num_classes).to(DEVICE)

    # Loss + Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)

    # Training loop
    epochs = 100
    best_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for Xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Eval on test set
        model.eval()
        correct = 0
        total = 0
        test_loss = 0

        with torch.inference_mode():
            for Xb, yb in test_loader:
                logits = model(Xb)
                loss = loss_fn(logits, yb)
                test_loss += loss.item() * Xb.size(0)

                preds = logits.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        test_loss /= len(test_loader.dataset)
        test_acc = correct / total * 100

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | TestLoss={test_loss:.4f} | Acc={test_acc:.2f}%")

    # Save model and label encoder
    os.makedirs(DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{DIR}/ocr_model.pth")
    joblib.dump(le, f"{DIR}/label_encoder.joblib")

    # Final Overall Model Evaluation
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # adjust batch_size if needed

    all_preds = []

    model.eval()
    with torch.inference_mode():  # prevents gradient computation
        for xb, _ in test_loader:
            xb = xb.cuda()  # move batch to GPU if needed
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()  # move predictions to CPU
            all_preds.append(preds)

    # Concatenate all batch predictions
    all_preds = np.concatenate(all_preds, axis=0)

    # Convert numeric labels back to original classes
    true_labels = le.inverse_transform(y_test.cpu().numpy())
    pred_labels = le.inverse_transform(all_preds)

    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title("Character Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=300)

    print("Training complete.")
