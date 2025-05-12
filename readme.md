# MNIST Classification Convolutional Neural Network

## Overview
This project implements a convolutional neural network for MNIST digit classification. The model is trained and evaluated using PyTorch and includes functionalities for checkpoint saving/loading, resuming training, and dynamic file management for improved organization and reusability.

## Functionalities

1. **Training the Model**
   - Modular training function that supports resumed training.
   - Tracks loss and accuracy metrics over epochs.

2. **Checkpoint System**
   - Save model state, optimizer state, loss, and accuracy every `n` epochs (customizable).
   - Default "best" checkpoint system for inference and resuming.
   - Flexible naming convention based on script name, model version, and training configuration.

3. **Evaluation**
   - Evaluate model performance using test datasets.
   - Currently only support for  single-inference from the test dataset.

4. **Dynamic Naming**
   - File naming convention includes script name and training configurations.
   - Automatically organizes checkpoints into specific directories.

5. **Custom Directory Management**
   - Create unique project directories for checkpoints and logs.
   - Prevents overwriting and ensures traceability.

## Future Improvements

- Add TensorBoard support for visualization.
- Integrate energy consumption analysis based on training metrics.
- Create a custom input method for inference.
