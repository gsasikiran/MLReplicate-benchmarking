import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training and validation losses
try:
    plt.figure()
    plt.plot(
        experiment_data["momentum_tuning"]["synthetic_dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["momentum_tuning"]["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plotting validation metrics
try:
    plt.figure()
    plt.plot(
        experiment_data["momentum_tuning"]["synthetic_dataset"]["metrics"]["val"],
        label="PWIS (Validation Metric)",
    )
    plt.title("Validation Metric (PWIS) for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_validation_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()
