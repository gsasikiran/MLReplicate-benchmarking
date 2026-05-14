import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot settings
types = [
    "regularization_no_regularization",
    "regularization_l2",
    "regularization_dropout",
]
dataset_name = "synthetic_worker_data"

for reg_type in types:
    try:
        plt.figure()
        plt.plot(
            experiment_data[reg_type][dataset_name]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data[reg_type][dataset_name]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{reg_type.replace('_', ' ').title()} - Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{reg_type}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {reg_type}: {e}")
        plt.close()

    # Optionally for predictions and ground truth
    try:
        plt.figure()
        plt.scatter(
            experiment_data[reg_type][dataset_name]["ground_truth"],
            experiment_data[reg_type][dataset_name]["predictions"],
            alpha=0.5,
        )
        plt.title(f"{reg_type.replace('_', ' ').title()} - Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"{reg_type}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {reg_type}: {e}")
        plt.close()
