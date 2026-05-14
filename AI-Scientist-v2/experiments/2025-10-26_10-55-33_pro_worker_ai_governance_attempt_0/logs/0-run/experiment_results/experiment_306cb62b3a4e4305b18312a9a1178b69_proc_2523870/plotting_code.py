import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    plt.figure()
    for batch_size, data in experiment_data["batch_size_tuning"].items():
        plt.plot(
            data["losses"]["train"],
            label=f"Train Loss - Batch {batch_size}",
            linestyle="-",
        )
        plt.plot(
            data["losses"]["val"],
            label=f"Validation Loss - Batch {batch_size}",
            linestyle="--",
        )
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot PWIS metric
try:
    plt.figure()
    for batch_size, data in experiment_data["batch_size_tuning"].items():
        plt.plot(data["metrics"]["val"], label=f"PWIS - Batch {batch_size}")
    plt.title("Validation PWIS by Batch Size")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_pwIs_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS curves plot: {e}")
    plt.close()
