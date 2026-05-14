import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
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
    epochs = np.arange(
        1, len(experiment_data["synthetic_worker_data"]["losses"]["train"]) + 1
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_worker_data"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_worker_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot predictions versus ground truth
try:
    plt.figure()
    plt.scatter(
        experiment_data["synthetic_worker_data"]["ground_truth"],
        experiment_data["synthetic_worker_data"]["predictions"],
        alpha=0.5,
    )
    plt.plot([0, 1], [0, 1], "r--")  # Ideal line
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "synthetic_worker_data_predictions_vs_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
