import sys
import numpy as np
import preprocessing as pre
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# DIR = "/windows/Users/thats/Documents/ocr-repo-files"
# DATA = "dataset2/Img"
DIR = r"/Users/dhruv/OneDrive/Desktop/4AL3/ocr-repo-files"
DATA = r"/Users/dhruv/OneDrive/Desktop/4AL3/ocr-repo-files/dataset2/Img"  # set directory path
FEATURE_DIR = f"{DIR}/features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# range of files to build training features from
first_train_file = 0
last_train_file = 3000
X, y = pre.get_same_length_features_and_labels(
        f"{FEATURE_DIR}/ordered_labels.npy", FEATURE_DIR,
            first_train_file, last_train_file)

y = np.array(y)

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train_feature_len = X_train.shape[1]

print(X_train_feature_len)

# convert y's letters to numerals
# y_train is a list of the labels corresponding to each features in X_train
# this encoder will convert those letters to numerical values used for training
# eg. a -> 0, b -> 1
le = LabelEncoder()
y_train_numeric = le.fit_transform(y_train)

##### Moved to train set creation above
# first_test_file = 1001
# last_test_file = 1501
# X_test, y_test = pre.get_same_length_features_and_labels(
#         f"{FEATURE_DIR}/ordered_labels.npy", FEATURE_DIR,
#             first_test_file, last_test_file)
# X_test_feature_len = X_test.shape[1]

# create label encoders for test set as well
# use the same letters as training set?
y_test_numeric = le.transform(y_test)

# print(y_test_numeric)

# doing some research, linear with 297432 features will explode my computer
# reduce features? or just dont use a linear model
pca = PCA(n_components=0.99, svd_solver='full')
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Reduced feature size: {X_train.shape[1]}")

# convert to tensors, both test and train
# i.e. [array([1,2,3]), array([4,5,6])] -> tensor([[1,2,3],[4,5,6]])
X_train = torch.from_numpy(X_train).type(torch.float) 
y_train = torch.from_numpy(y_train_numeric).type(torch.long)

X_test = torch.from_numpy(X_test).type(torch.float) 
y_test = torch.from_numpy(y_test_numeric).type(torch.long)


# exit()

# attach to a whatever device is avaliable
X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

# since we are using HOG features, we can use a simple NN for now?
# model class
class NNmodel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # hidden layers/neurons
        # for now basic linear layers
        self.layer_1 = nn.Linear(in_size, 350) # how to determine hidden neurons?
        self.layer_2 = nn.Linear(350, 200)
        self.layer_3 = nn.Linear(200, out_size)
        # initialize dropout layer, helps prevent overfitting i believe
        # kinda forgot how this works need to review
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        # the forward pass is for now just relu activations and dropout
        # note to self: relu essentially just converts negative values to 0
        x = self.dropout(torch.relu(self.layer_1(x))) #first go through layer 1, compute relu + dropout
        x = self.dropout(torch.relu(self.layer_2(x))) #go through layer 2
        x = self.layer_3((x)) # is relu needed here? 
        return x
    
# initialize model
num_labels = len(np.unique(y_train_numeric)) # number of unique labels
model = NNmodel(X_train.shape[1], num_labels).to(DEVICE)

# loss function and optimizer?
# Use a cross entropy loss for now, good to use for multi class classification which is what we are doing
loss_fn = nn.CrossEntropyLoss()

# adam optimizer? just use for now
# who tf is adam???
# need to look into more
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# nvm i hate adam lets use sgd
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



# training loooooooooooooooooop
# should feed in batches but for now just do all at once
epochs = 10000
for epoch in range(epochs):
    model.train()
    # forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # backwards pass
    optimizer.zero_grad() # zero the gradients
    loss.backward() # back propagation
    optimizer.step() # update weights

    # testing loop?

    model.eval() # set model to eval mode
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)
        # add acccuracy calculation?
        # calculate accuracy
        preds = torch.argmax(test_pred, dim=1)
        correct = (preds == y_test).sum().item()
        total = y_test.size(0)
        test_acc = correct / total * 100

    # print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# save model
torch.save(model.state_dict(), f"{DIR}/ocr_model.pth")