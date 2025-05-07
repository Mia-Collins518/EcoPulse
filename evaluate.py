import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import NeuralNetwork
import random
import numpy as np

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Download test data from open datasets.
test_data = datasets.MNIST(
   root="data",
   train=False,
   download=True,
   transform=ToTensor(),
)

# Create data loader. 
test_dataloader = DataLoader(test_data, batch_size=64)

# If a GPU is available, use it; otherwise, fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Evaluation Function
def test(dataloader, model, loss_fn):
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   model.eval()
   test_loss, correct = 0, 0
   with torch.no_grad():
       for batch_idx, (X, y) in enumerate(dataloader):
           X, y = X.to(device), y.to(device)
           pred = model(X)
           test_loss += loss_fn(pred, y).item()
           correct += (pred.argmax(1) == y).type(torch.float).sum().item()

   test_loss /= num_batches
   correct /= size
   print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()

# Run evaluation
test(test_dataloader, model, loss_fn)

import random

# Single inference example with random selection
random_index = random.randint(0, len(test_data) - 1)  # Get a random index
x, y = test_data[random_index][0], test_data[random_index][1]  # Get the data and label at that index

# Run inference
with torch.no_grad():
    x = x.to(device).unsqueeze(0)  # Add batch dimension
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')