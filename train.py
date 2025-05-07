import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NeuralNetwork

# Hyper parameters
batch_size = 50
epochs = 5

# Download training data from open datasets.
training_data = datasets.MNIST(
   root="data",
   train=True,
   download=True,
   transform=ToTensor(),
)

# Create data loader.
train_dataloader = DataLoader(training_data, batch_size=batch_size)

# If an accelerator is available, it will be used, else the CPU will be used.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initalize model, loss function, optimizer.
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# TensorBoard writer
writer = SummaryWriter()

# Training function
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Calculate accuracy manually
        _, predicted = torch.max(pred, 1)  # Get the index of the max log-probability
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
        total_accuracy += accuracy

        # Log loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Log the average loss and accuracy for the epoch
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_accuracy, epoch)

# Main Training Loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t+1)

print("Training Complete!")
writer.flush()

# Save Model Weights
torch.save(model.state_dict(), "model_weights.pth")
print("Saved PyTorch Model State to model_weights.pth")

# Close the TensorBoard writer
writer.close()