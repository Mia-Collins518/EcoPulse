import torch
import os

# This function saves metadata from models that have been trained in the checkpoint directory. It takes the epoch, model, optimizer, loss, accuracy...
def save_checkpoint(epoch, model, optimizer, loss, accuracy, model_type, training_version, save_dir):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Default path for the best model
    best_model_path = os.path.join(save_dir, "best_model.pth.tar")

    # Load the current best accuracy if the best model file exists
    best_accuracy = 0.0  # Default value
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        best_accuracy = checkpoint.get('val_accuracy', 0.0)

    # Check if this is the best model
    is_best = accuracy > best_accuracy
    if is_best:
        best_accuracy = accuracy

    # Save the current checkpoint
    checkpoint_name = f"{model_type}_{training_version}_Epoch{epoch}.pth.tar"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    # Prevents files from being overwritten 
    if os.path.exists(checkpoint_path):
        raise Exception("File already exists")

    # Save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_accuracy': accuracy
    }, checkpoint_path)

    # Save as the best model if applicable
    if is_best:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'val_accuracy': accuracy
        }, best_model_path)
        print(f"New best model saved: {best_model_path}")

    print(f"Checkpoint saved: {checkpoint_path}")

# This function loads only the model's weights to be used for inference. It takes the model as input. It also can optionally take checkpoint_dir and device (default set
# to cpu) as input. If a specific checkpoint is needed for inference, then it can be inputted when the function is called. Otherwise, the default is the best model path. 
def load_checkpoint_for_inferece(model, save_dir, device='cpu'):

    best_model_path = os.path.join(save_dir, "best_model.pth.tar")

    # Check if the checkpoint exists
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No best model saved yet at: {best_model_path}") # Throws an error if the file path dosen't exist
    
    # Load the checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)

    # Load the model state (the weights)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load accuracy for display
    accuracy = checkpoint.get('val_accuracy', None)

    print(f"Model weights loaded from {best_model_path}, Accuracy: {accuracy:.2f}%")

# This function loads all of the stored metadata from the checkpoint file so that trainnig can continue from the latest epoch. It takes the input model, optimzer,
# checkpoint_path, and device (default set to cpu). The checkpoint_path is changable, with it being defined globally in the train script (default set to project_dir/best_model.pth.tar)
def load_checkpoint_training(model, optimizer, checkpoint_path, device='cpu'):
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load states into model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract metadata
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    accuracy = checkpoint.get('val_accuracy', None)

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming training from epoch {epoch}, loss {loss:.4f}, and accuracy {accuracy:.2f}%")

    return epoch, loss, accuracy
