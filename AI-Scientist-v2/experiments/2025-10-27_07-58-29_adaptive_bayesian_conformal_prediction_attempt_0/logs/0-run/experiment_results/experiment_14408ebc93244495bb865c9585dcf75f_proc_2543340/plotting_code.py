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

# Plot training vs validation loss
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot predictions vs ground truth for validation
try:
    plt.figure()
    plt.scatter(
        experiment_data["synthetic_data"]["ground_truth"],
        experiment_data["synthetic_data"]["predictions"][0],
        label="Predictions",
        alpha=0.5,
    )
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
