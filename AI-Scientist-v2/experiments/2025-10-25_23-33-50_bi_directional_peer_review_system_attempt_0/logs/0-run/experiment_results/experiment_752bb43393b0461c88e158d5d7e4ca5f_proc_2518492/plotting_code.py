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
    plt.plot(experiment_data["peer_review"]["losses"]["train"], label="Training Loss")
    plt.plot(experiment_data["peer_review"]["losses"]["val"], label="Validation Loss")
    plt.title("Loss Curves for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

# Plot RQI over epochs
try:
    plt.figure()
    plt.plot(
        experiment_data["peer_review"]["metrics"]["train"],
        label="Review Quality Index (RQI)",
    )
    plt.title("RQI Over Epochs for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("RQI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_rqi.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQI plot: {e}")
    plt.close()

# Plot predictions vs ground truth
try:
    plt.figure()
    plt.scatter(
        experiment_data["peer_review"]["ground_truth"],
        experiment_data["peer_review"]["predictions"],
    )
    plt.title("Predictions vs Ground Truth for Peer Review Dataset")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "peer_review_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
