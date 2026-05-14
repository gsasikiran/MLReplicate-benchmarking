import matplotlib.pyplot as plt
import numpy as np
import os

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
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot validation metrics
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["metrics"]["val"],
        label="Validation Metric (Mean WIS)",
    )
    plt.title("Validation Metrics over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean WIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_validation_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()
