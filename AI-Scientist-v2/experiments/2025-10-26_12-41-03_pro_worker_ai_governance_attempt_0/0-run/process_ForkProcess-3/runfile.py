import os
import numpy as np

# Load the experiment data from the working directory
experiment_data = np.load(
    os.path.join(os.getcwd(), "working", "experiment_data.npy"), allow_pickle=True
).item()

# Extract metrics for the synthetic dataset
dataset_name = "synthetic_data"
train_losses = experiment_data[dataset_name]["losses"]["train"]
val_losses = experiment_data[dataset_name]["losses"]["val"]

# Print the metrics with clear labels
print(f"Dataset: {dataset_name}")
if train_losses:
    print("Final training loss:", train_losses[-1])
if val_losses:
    print("Final validation loss:", val_losses[-1])
