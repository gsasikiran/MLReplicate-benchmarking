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

# Plot training and validation metrics for synthetic dataset
try:
    epochs = np.arange(
        1, len(experiment_data["synthetic_data"]["metrics"]["train"]) + 1
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_data"]["metrics"]["train"],
        label="Train Metric",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_data"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("Training and Validation Metrics - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "metrics_synthetic.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# Plot training and validation losses for synthetic dataset
try:
    plt.figure()
    plt.plot(
        epochs, experiment_data["synthetic_data"]["losses"]["train"], label="Train Loss"
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "losses_synthetic.png"))
    plt.close()
except Exception as e:
    print(f"Error creating losses plot: {e}")
    plt.close()
