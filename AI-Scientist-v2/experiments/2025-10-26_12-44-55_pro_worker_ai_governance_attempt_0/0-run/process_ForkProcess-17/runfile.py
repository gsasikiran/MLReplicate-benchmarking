import os
import numpy as np

# Load the experiment data
experiment_data = np.load(
    os.path.join(os.getcwd(), "working", "experiment_data.npy"), allow_pickle=True
).item()

# Extract and print metrics
dataset_name = "synthetic_worker_data"
print(f"Dataset: {dataset_name}")

# Training metrics
final_train_loss = experiment_data["loss_function_ablation"][dataset_name]["losses"][
    "train"
][-1]
print(f"Final training loss: {final_train_loss:.4f}")

# Validation metrics
final_val_loss = experiment_data["loss_function_ablation"][dataset_name]["losses"][
    "val"
][-1]
final_val_WIS = experiment_data["loss_function_ablation"][dataset_name]["metrics"][
    "val"
][-1]
print(f"Final validation loss: {final_val_loss:.4f}")
print(f"Final validation WIS: {final_val_WIS:.4f}")
