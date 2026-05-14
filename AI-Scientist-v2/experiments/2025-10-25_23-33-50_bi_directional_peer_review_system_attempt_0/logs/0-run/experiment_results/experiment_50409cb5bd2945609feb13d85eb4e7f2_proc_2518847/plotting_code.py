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
    # Training and Validation Losses
    plt.figure(figsize=(12, 6))
    for scaling_method in experiment_data["input_feature_scaling"]:
        plt.plot(
            experiment_data["input_feature_scaling"][scaling_method]["losses"]["train"],
            label=f"Train {scaling_method}",
        )
        plt.plot(
            experiment_data["input_feature_scaling"][scaling_method]["losses"]["val"],
            label=f"Validation {scaling_method}",
            linestyle="--",
        )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Predictions vs Ground Truth
    for scaling_method in experiment_data["input_feature_scaling"]:
        plt.figure()
        plt.scatter(
            experiment_data["input_feature_scaling"][scaling_method]["ground_truth"],
            experiment_data["input_feature_scaling"][scaling_method]["predictions"],
        )
        plt.plot([0, 1], [0, 1], "r--")  # Line for ideal prediction
        plt.title(f"Predictions vs Ground Truth ({scaling_method})")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(
                working_dir, f"predictions_vs_ground_truth_{scaling_method}.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating predictions plot for {scaling_method}: {e}")
    plt.close()
