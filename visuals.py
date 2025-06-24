from tensorboardX import SummaryWriter
import os
from ptflops import get_model_complexity_info

# Initializes a TensorBoard writer inside a specific project folder that corresponds to the checkpoints
def get_tensorboard_writer(project_name):
    log_dir = os.path.join("runs", project_name)
    return SummaryWriter(log_dir=log_dir)

# Logs training metrics to TensorBoard
def log_training_metrics(writer, epoch, loss, accuracy):
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

"""
# Gets the number of multiply-accumulate operations (MACs = multiplication followed by an addition) and the parameters of the model
def log_flops(writer, model):
    macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True) # Gets MACs and parameters from 1 grayscale image of 28x28 size (MNIST digit)
    writer.add_scalar("FLOPs/model", (macs*2), 0)
    writer.add_scalar("Parameters/model", params, 0)
    return macs
    #print(f"FLOPs (MACs): {macs}, Params: {params}")
"""
# Logs the time it takes to train over epochs
def log_time(writer, elapsed_seconds, epoch):
    writer.add_scalar("Time/train", elapsed_seconds, epoch)

# Energy conversion based on the FLOPs of a model
def flops_to_energy(macs):
    flops = macs * 2                       # MAC → FLOPs
    joules_per_flop = 1e-12                # J per FLOP
    joules = flops * joules_per_flop
    kwh = joules / (3600 * 1000)           # 1 kWh = 3.6e6 J
    return joules, kwh

def log_energy(writer, epoch, kwh):
    print("fix")

# Optionally flush and close the writer
def close_writer(writer):
    writer.flush()
    writer.close()