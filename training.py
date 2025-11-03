import sys
import numpy as np
import preprocessing as pre
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder

DIR = "/windows/Users/thats/Documents/ocr-repo-files"
DATA = "dataset2/Img"
FEATURE_DIR = f"{DIR}/features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# range of files to build training features from
first_train_file = 0
last_train_file = 100
X_train, y_train = pre.get_same_length_features_and_labels(
        f"{FEATURE_DIR}/ordered_labels.npy", FEATURE_DIR,
            first_train_file, last_train_file)
X_train_feature_len = X_train.shape[1]

# convert y's letters to numerals
le_train = LabelEncoder()
y_train_numeric = le_train.fit_transform(y_train)

# range of files to build test features from
first_test_file = 101
last_test_file = 120
X_test, y_test = pre.get_same_length_features_and_labels(
        f"{FEATURE_DIR}/ordered_labels.npy", FEATURE_DIR,
            first_test_file, last_test_file)
X_test_feature_len = X_test.shape[1]
# print(X_test_feature_len)
# sys.exit()

le_test = LabelEncoder()
y_test_numeric = le_test.fit_transform(y_test)

X_train = torch.from_numpy(X_train).type(torch.float) # convert to tensors
y_train = torch.from_numpy(y_train_numeric).type(torch.float)

X_test = torch.from_numpy(X_test).type(torch.float) # convert to tensors
y_test = torch.from_numpy(y_test_numeric).type(torch.float)

class LetterPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=X_train_feature_len,
                                    out_features=50000)
        self.layer_2 = nn.Linear(in_features=50000, out_features=1)
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) 

model_0 = LetterPredictor().to(DEVICE)

# Make predictions with the model
untrained_preds = model_0(X_test.to(DEVICE))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")
