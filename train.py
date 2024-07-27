import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from model.model import SimpleNeuralNetwork
from sklearn.model_selection import train_test_split

# training args
DATASET_FILE = "dataset.json"   # path to dataset
TRAIN_TEST_SPLIT_SIZE = 0.2     # 20% of the dataset for validation, 80% for training
LEARNING_RATE = 0.005           # amount the weights should be ajusted over each epoch
N_EPOCHS = 50000                # number of iterations over the dataset
LOSS_LOGGING_RATE = 20          # log loss all 20 epochs

# initialize model
model = SimpleNeuralNetwork(n_inp=8064, n_out=10) # a 84x96 pixel image has 8064 pixels and there are 10 numbers to predict

if (os.path.exists("model.pth")): model.load_state_dict(torch.load("model.pth"))

# load dataset
with open(DATASET_FILE, "r") as file:
    dataset = json.loads(file.read())

# Extract features and labels
features = torch.stack([torch.Tensor(data["pixels"]).view(-1) for data in dataset])
labels = torch.Tensor([data["num"] for data in dataset]).long()

# logging for debug
print(f"Feature tensor shape: {features.shape}")
print(f"Label tensor shape: {labels.shape}")

# Generate train and test split
X_train, X_test, y_train, y_test = train_test_split(features.numpy(), labels.numpy(), test_size=TRAIN_TEST_SPLIT_SIZE) # 20% for test

# Convert the split data back to PyTorch tensors
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train).long()
y_test = torch.Tensor(y_test).long()

# logging for debug
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# training loop
try:
    for epoch in range(0, N_EPOCHS):
        pred_Y = model(X_train) # make prediction

        loss = criterion(pred_Y, y_train) # calculate loss between prediction and expected

        # log loss
        if epoch % LOSS_LOGGING_RATE == 0:
            print(f"Epoch: {epoch}/{N_EPOCHS}; Loss: {loss}")

        # do backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
except KeyboardInterrupt:
    print("Detected Keyboard Interrupt! Stopping training...")

# Evaluate model on test split
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

    print(f"Test loss: {loss}")

torch.save(model.state_dict(), "model.pth")
