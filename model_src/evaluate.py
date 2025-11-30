import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from CNN import NeuralNetwork
import random
import numpy as np
import os
from checkpoint_utils import load_checkpoint_for_inferece

digets = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] # For output
batch_size = 50 # For controling the data loader

# Download test data from the open MNIST dataset and download as tensors.
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

# Add project variable
project = "main"  # Same project name as in train.py
project_dir = os.path.join("checkpoints", project)

# Create data loader. 
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# If a GPU is available, use it; otherwise, fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNetwork().to(device) # Load model from model.py
load_checkpoint_for_inferece(model, project_dir) # load weights using the load_checkpoint_for_inferece in utils.py (default best_model_path)
model.eval() # Puts model in eval mode

# Runs a single inference. Takes input of the model, input tensor (X), and the device.
def single_inference(model, X, device):
   with torch.no_grad():
      X = X.to(device).unsqueeze(0)  # Add batch dimension to the tensor so it can be fed into the model
      pred = model(X) # Get the model's prediction
      return(pred)
   
# Single inference example with random selection
random_index = random.randint(0, len(test_data) - 1)  # Get a random index
x, y = test_data[random_index][0], test_data[random_index][1]  # Get the data and label at that index

# Compare predicted/actual to the output array of digets and result is script output 
predicted, actual = digets[single_inference(model, x, device)[0].argmax(0)], digets[y] 
print(f'Predicted: "{predicted}", Actual: "{actual}"')