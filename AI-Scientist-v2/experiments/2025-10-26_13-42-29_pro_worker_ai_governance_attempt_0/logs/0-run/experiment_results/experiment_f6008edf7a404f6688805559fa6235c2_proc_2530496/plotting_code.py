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
for scaling_type in experiment_data["input_feature_scaling"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["input_feature_scaling"][scaling_type]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["input_feature_scaling"][scaling_type]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{scaling_type.capitalize()} - Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{scaling_type}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {scaling_type} losses: {e}")
        plt.close()

# Plot predictions vs ground truth
for scaling_type in experiment_data["input_feature_scaling"]:
    try:
        plt.figure()
        plt.scatter(
            experiment_data["input_feature_scaling"][scaling_type]["ground_truth"],
            experiment_data["input_feature_scaling"][scaling_type]["predictions"],
            alpha=0.5,
        )
        plt.title(f"{scaling_type.capitalize()} - Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.axline((0, 0), slope=1, color="red", linestyle="--")  # Line y=x
        plt.savefig(
            os.path.join(working_dir, f"{scaling_type}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating plot for {scaling_type} predictions vs ground truth: {e}"
        )
        plt.close()
