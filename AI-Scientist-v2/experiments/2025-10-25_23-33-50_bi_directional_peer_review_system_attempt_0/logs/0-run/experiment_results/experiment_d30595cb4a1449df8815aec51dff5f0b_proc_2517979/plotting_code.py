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

# Plotting training and validation loss
try:
    plt.figure()
    plt.plot(
        experiment_data["peer_review_quality"]["metrics"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["peer_review_quality"]["metrics"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "peer_review_quality_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

# Plotting ground truth vs predictions
try:
    plt.figure()
    plt.scatter(
        experiment_data["peer_review_quality"]["ground_truth"],
        experiment_data["peer_review_quality"]["predictions"],
        alpha=0.5,
    )
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.title("Ground Truth vs Predictions")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(
        os.path.join(working_dir, "peer_review_quality_ground_truth_vs_predictions.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
