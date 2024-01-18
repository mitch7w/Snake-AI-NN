import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from matplotlib import pyplot as plt

# import datasets
game_states = np.load('saved_game_states.npy')
game_scores = np.load('saved_game_scores.npy')

# Convert the NumPy array to a PyTorch tensor
game_states_tensor = torch.from_numpy(
    game_states).long()  # Use long type for indices
one_hot_grid = torch.nn.functional.one_hot(
    game_states_tensor, num_classes=4)  # Apply one-hot encoding
# Reshape the one-hot encoding to match the desired shape [14976, 20, 20, 4]
one_hot_grid = one_hot_grid.permute(0, 2, 3, 1)
one_hot_grid = one_hot_grid.float()  # Convert to float
# Now, one_hot_grid can be used as input to your neural network
X_train_tensor = one_hot_grid
print(X_train_tensor.shape)
y_train_tensor = torch.from_numpy(game_scores).float()

# Create a TensorDataset from the input and output tensors
training_data = TensorDataset(X_train_tensor, y_train_tensor)
# test_data = TensorDataset(X_train_tensor, y_train_tensor)

batch_size = 256

# Create data loaders.
train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*20*4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_values = []


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        loss_values.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# show loss curve
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth"))


# TODO maybe CNN
# TODO fix input one hot encoding and stuff like that
# TODO change up completely and just get it to learn via score or move on to image datasets/other example tutorials
