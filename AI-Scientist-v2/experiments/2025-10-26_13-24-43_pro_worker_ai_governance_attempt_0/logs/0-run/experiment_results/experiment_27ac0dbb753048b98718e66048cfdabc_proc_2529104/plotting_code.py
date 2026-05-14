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
        experiment_data["synthetic_data"]["losses"]["train"], label="Training Loss"
    )
    plt.title("Training Loss Curve for Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.scatter(
        experiment_data["synthetic_data"]["ground_truth"],
        experiment_data["synthetic_data"]["predictions"],
        alpha=0.5,
    )
    plt.title("Ground Truth vs Predictions for Synthetic Data")
    plt.xlabel("Ground Truth PWIS")
    plt.ylabel("Predicted PWIS")
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_ground_truth_vs_predictions.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
