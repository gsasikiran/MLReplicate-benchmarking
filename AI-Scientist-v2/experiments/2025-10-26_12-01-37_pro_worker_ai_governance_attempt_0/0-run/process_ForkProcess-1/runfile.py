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

try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["train"], label="Train Loss"
    )
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.scatter(
        experiment_data["synthetic_dataset"]["ground_truth"],
        experiment_data["synthetic_dataset"]["predictions"],
    )
    plt.title("Ground Truth vs Predictions")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.plot([0, 1], [0, 1], "r--")  # diagonal line
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_ground_truth_vs_predictions.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
