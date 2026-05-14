import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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
        experiment_data["early_stopping"]["synthetic_dataset"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["early_stopping"]["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot additional metrics if available
try:
    metrics = experiment_data["early_stopping"]["synthetic_dataset"]["metrics"]
    plt.figure()
    plt.plot(metrics["train"], label="Train Metric")
    plt.plot(metrics["val"], label="Validation Metric")
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
