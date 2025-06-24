import torch
from torch import nn
from tensorboardX import SummaryWriter # Currently not utilized in code, planning on revamping the way I track the data, will change in future. 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from CNN import NeuralNetwork, get_name
from checkpoint_utils import save_checkpoint, load_checkpoint_training
from visuals import get_tensorboard_writer, log_training_metrics, log_time, flops_to_energy
import os 
from ptflops import get_model_complexity_info
import time

# Hyper parameters
batch_size = 50
epochs = 35
learn_rate = 1e-3

# Saving/Loading Parameters (Customizable functionaility based on the following variables)
resume_training = True # False = Start training a new model, True = resume training on an existing model by loading it's data
save_checkpoint_num = 5 # Defines how often to save models (to avoid overwhelm). 0 = off (no saving at all)
model_name = get_name() # Gets the name of the model script

# This defines the iteration of the training script, allowing seemless saving capabilities as I improve the training method (change 1.0 to higher version indicators)
train_version = f"{os.path.splitext(os.path.basename(__file__))[0]}1.0"

# Add project variable
project = "test_2"  # Customize this for specific projects
project_dir = os.path.join("checkpoints", project)  # Creates project-specific checkpoint directory in the checkpoints directory

load_checkpoint_path = os.path.join(project_dir, "best_model.pth.tar") # Default is project_dir/best_model.pth.tar to load best model, but any existing model can be loaded


# If an accelerator is available, it will be used, else the CPU will be used.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train(dataloader, model, loss_fn, optimizer, epoch, initial_loss=0.0, initial_accuracy=0.0):
    size = len(dataloader.dataset)
    model.train()
    total_loss = initial_loss
    total_accuracy = initial_accuracy

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
        _, predicted = torch.max(pred, 1)
        total_accuracy += (predicted == y).sum().item() / y.size(0)

        # Log loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Log the average loss and accuracy for the epoch
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = (total_accuracy / len(dataloader))*100
    return avg_loss, avg_accuracy

def training_loop(model, optimizer, loss_fn, train_dataloader, epochs, start_epoch, avg_loss=0.0, avg_accuracy=0.0):
    for epoch in range(start_epoch, epochs):
        # Epoch logging uses 1-indexing for readability
        display_epoch = epoch + 1
        print(f"\nEpoch {display_epoch}\n-------------------------------")
        
        # Train the model for the current epoch
        avg_loss, avg_accuracy = train(train_dataloader, model, loss_fn, optimizer, display_epoch)
        print(f"Epoch {display_epoch} Summary: Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}%")
        
        log_training_metrics(writer, epoch, avg_loss, avg_accuracy)

        elapsed = time.time() - start_time
        log_time(writer, elapsed, epoch)

        # if save_checkpoint_num var is 0, there will be no saving at all.
        if save_checkpoint_num != 0:
            # Save Model Weights every 5 epochs
            if display_epoch % save_checkpoint_num == 0:
                save_checkpoint(display_epoch, model, optimizer, avg_loss, avg_accuracy, model_name, train_version, project_dir)

if __name__ == "__main__":
    # Initalize model, loss function, optimizer.
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    writer = get_tensorboard_writer(project) # create a tensorboard writer
    macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=False) # Gets MACs and parameters from 1 grayscale image of 28x28 size (MNIST digit)
    
    # static model metadata (epoch 0)
    writer.add_scalar("FLOPs/model", (macs*2), 0)
    writer.add_scalar("Parameters/model", params, 0)

    # energy both in joules and kWh
    joules, kwh = flops_to_energy(macs)
    #writer.add_scalar("Energy/train_joules", joules, 0)
    writer.add_scalar("Energy/train_kwh",   kwh,    0)

    # learning rate as a static scalar
    writer.add_scalar("Hyperparams/learning_rate", learn_rate, 0)
    
    # Download training data from open datasets.
    training_data = datasets.MNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    )

    # Create data loader.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    # Intailizing loadable variables
    start_epoch = 0
    start_time = time.time()
    initial_loss = 0.0
    initial_accuracy = 0.0

    # Throws an error if resume training is on but a new project directory is being saved to
    if (resume_training == True) & (os.path.exists(project_dir) == False):
        raise Exception("Resume training impossible with current settings, change the resume_training or epochs variable")


    # Resume training if enabled
    if resume_training == True:
        # Loads epoch, loss, and accuracy from checkpoint
        load_epoch, load_loss, load_acc = load_checkpoint_training(model, optimizer, load_checkpoint_path, device)

        # Throws an error if the epochs variable is lesss then or equal to the existing training epoch
        if load_epoch >= epochs:
            raise ValueError("Epochs variable too small to continue training.")

        start_epoch = load_epoch
        print(f"Resuming training from epoch {start_epoch}...")

    # Main Training Loop
    training_loop(model, optimizer, loss_fn, train_dataloader, epochs, start_epoch, initial_loss, initial_accuracy)