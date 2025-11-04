import sys
import numpy as np
import preprocessing as pre
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder

# DIR = "/windows/Users/thats/Documents/ocr-repo-files"
# DATA = "dataset2/Img"
DIR = r"/Users/dhruv/OneDrive/Desktop/4AL3/ocr-repo-files"
DATA = r"/Users/dhruv/OneDrive/Desktop/4AL3/ocr-repo-files/dataset2/Img"  # set directory path
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
le = LabelEncoder()
y_train_numeric = le.fit_transform(y_train)

# range of files to build test features from
first_test_file = 101
last_test_file = 120
X_test, y_test = pre.get_same_length_features_and_labels(
        f"{FEATURE_DIR}/ordered_labels.npy", FEATURE_DIR,
            first_test_file, last_test_file)
X_test_feature_len = X_test.shape[1]

# create label encoders for test set as well
# use the same letters as training set?
y_test_numeric = le.fit_transform(y_test)

# convert to tensors, both test and train
X_train = torch.from_numpy(X_train).type(torch.float) 
y_train = torch.from_numpy(y_train_numeric).type(torch.long)

X_test = torch.from_numpy(X_test).type(torch.float) 
y_test = torch.from_numpy(y_test_numeric).type(torch.long)

# since we are using HOG features, we can use a simple NN for now?
# model class
class NNmodel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # hidden layers/neurons
        # for now basic linear layers
        self.layer_1 = nn.Linear(in_size, 5)
        self.layer_2 = nn.Linear(5, 10)
        self.layer_3 = nn.Linear(10, out_size)
        # initialize some sort of dropout layer, helps prevent overfitting i believe
        # kinda forgot how this works need to review
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        # the forward pass is for now just relu activations and dropout
        x = self.dropout(torch.relu(self.layer_1(x)))
        x = self.dropout(torch.relu(self.layer_2(x)))
        x = self.layer_3(x)
        return x
    
# initialize model
num_labels = len(np.unique(y_train_numeric)) # number of unique labels
model = NNmodel(X_train_feature_len, num_labels).to(DEVICE)

# loss function and optimizer?
# Use a cross entropy loss for now, good to use for multi class classification which is what we are doing
loss_fn = nn.CrossEntropyLoss()

# adam optimizer? just use for now
# who tf is adam???
# need to look into more
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# nvm i hate adam lets use sgd
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# attach to a whatever device is avaliable
X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
X_test, y_test = X_train.to(DEVICE), y_train.to(DEVICE)

# training loooooooooooooooooop
epochs = 100
for epoch in range(epochs):
    # forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # backwards pass
    optimizer.zero_grad() # zero the gradients
    loss.backward() # back propagation
    optimizer.step() # update weights

    # testing loop?

    # print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {0:.2f}% | Test Loss: {0:.5f}, Test Acc: {0:.2f}%")


# save model
