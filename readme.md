# EcoPulse: AI Model and Website Deployment 

## Overview
This project, named EcoPulse, is a full-stack application that deploys a convolutional neural network (CNN) for MNIST digit classification via a FastAPI backend and a Next.js (React/TypeScript) frontend. The model is trained and evaluated using PyTorch and includes functionalities for checkpoint saving/loading, resuming training, and dynamic file management for improved organization and reusability. The core focus is on not only model performance but also monitoring and quantifying the computational cost of the AI lifecycle, specifically through FLOPs and estimated energy consumption.

## Key Functionalities

1. **Training the Model**
   - Model Architecture: Implements a CNN based on the LeNet architecture for 10-class MNIST digit classification. 
   - Model Training: The training loop is a method that accepts parameters that support continued training. 
   - Uses PyTorch's ```bash torchvision``` to download and manage the MNIST dataset for training and evaluation.
     

2. **Checkpoint and Saving System**
   - Saving Logic: Checkpoints including model state, optimizer state, loss, and accuracy are saved on a customizable interval.
   - Dynamic Directory Management: checkpoints are automatically organized into specific project directories (e.g., ```bash checkpoints/project_name/```) to prevent overwriting and allow easy comparison of different runs.
      - Flexible naming convention based on script name, model version, and training configuration.
   - Flexible Resuming and Inferece: ```bash load_checkpoint_for_ training```function loads all necessary metadata to resume training runs, and ```bash load_checkpoint_for_ inference``` function defaults to loading the "best_model" but can load any specified checkpoint for the training process.
   - This functionality is intended to prevent overwriting of data and allow easy comparison of different runs. 
     
3. **Meterics WorkFlow**
   - Metrics Tracking: logs of traditional metrics (loss, accuracy) alongside additional non-traditional metrics: Model Parameters, FLOPs, Training Time, and Estimated Energy Consumption. 
   - TensorBoard Integration: uses TensorBoard to log all scalars during the training process.
   - Data Conversion: The ```bash convert.py``` script utilizes ```bash tbparse``` to automatically transform the ```bash .tfevent``` files into CSV files, allowing the frontend to utilize the data for display.

4. **Full-Stack API Integration**
   - FastAPI: The API framework serves as the bridge between the backend ML model and the frontend web interface.
   - Metric Discovery: API routes (```bash /projects```, ```bash /tags```) dynamically read the metrics directory, providing the frontend with a list of available training runs and metrics without hardcording.
   - Random Inference: The ```bash /random-inference``` endpoint loads the trained model, randomly selects a test image, runs the prediction, Base64-encodes the input images as a PNG, and returns the actual, predicted, and image data in a single JSON payload.
   - Custom Inference: The ```bash /custom-inference``` adds a POST API endpoint to handle upload Base64 image from the frontend, then returns the input image sas a PNG, the prediction in a single JSON payload.
        - Furture Improvements: The image processing on the image processing dosen't work very well. 


## Run Website: Local Deployment

1. **Start the Backend API**
   - Ensure you have Python 3.9+ installed.

   **Instructions**
   1. Open the first terminal window/tab.
   2. Activate the Python Virtual Environment (run these commands from the project root: EcoPulse_Project/):
   ```bash
   python3 -m venv .venv               # Create environment (only needs to be run once)
   source .venv/bin/activate           # Activate the environment (macOS/Linux)
   # OR: .venv\Scripts\activate          # Activate the environment (Windows)
   ```
   3. Install Python Dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
   4. Start the Backend Server:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   - Sucessful Outcome: you will see the message ```bash Uvicorn running on http://127.0.0.1:8000```
   - Keep this terminal window open and running.

2. **Start the Frontend App (Next.js)**
   - The frontend displays the user interface and utilizes data from the backend API. 

    **Instructions**
   1. Open a second terminal window/tab.
   2. Navigate tot the Frontend Directory:
   ```bash cd frontend ```
   3. Install Node Dependencies:
   ```bash
   npm install
   # OR: yarm install
   ```
   4. Start the frontend development server: 
   ```bash
   npm run dev
   # OR: yarn dev
   ```
   5. View the Website: Once the process completes, open your web browser to:
   6. ```bash http://localhost:3000```
  
## Website Shut Down
   - Since the website is running as two separate processes, you must stop them individually.

   1. Stop the Frontend Server: Go to the terminal window running ```bash npm run dev``` and press ```bash CTRL + C```.
   2. Stop the Backend Server: Go to the terminal window running ```bash uvicorn backend.main:app``` and press ```bash CTRL + C```.

   
