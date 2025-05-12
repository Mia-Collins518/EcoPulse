import torch
from torch import nn
import os

# Defines a neural network model for MNIST classification.
class NeuralNetwork(nn.Module):  # Inherits from nn.Module, the base class for all neural networks in PyTorch
    def __init__(self):
        super().__init__()  # Initialize the parent class
        # Layer to flatten input tensors (the MNIST 28x28 images) into 1D vectors
        self.flatten = nn.Flatten()
        
        # Linear layers with ReLU activations
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # First fully connected layer: input size 28*28 (784), output size 512
            nn.ReLU(),              # Apply ReLU activation function
            nn.Linear(512, 512),    # Second fully connected layer: input size 512, output size 512
            nn.ReLU(),              # Apply ReLU activation function
            nn.Linear(512, 10)      # Final fully connected layer: input size 512, output size 10 (number of classes)
        )

    # Define the forward pass (how data flows through the network)
    def forward(self, x):
        x = self.flatten(x) # Flatten the input tensor to a 1D vector
        logits = self.linear_relu_stack(x) # Pass the flattened tensor through the linear-ReLU stack
        return logits # Output raw scores (logits) for each class

# Gets the script name CNN from the file name CNN.py for checkpoint path naming   
def get_name():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    return(script_name)  