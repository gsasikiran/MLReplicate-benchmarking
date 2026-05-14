import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for scaling_name in experiment_data["feature_scaling"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["feature_scaling"][scaling_name]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["feature_scaling"][scaling_name]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{scaling_name.capitalize()} Scaling: Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{scaling_name}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {scaling_name} losses: {e}")
        plt.close()

    try:
        plt.figure()
        val_predictions = np.array(
            experiment_data["feature_scaling"][scaling_name]["predictions"]
        )[-1]
        ground_truth = np.array(
            experiment_data["feature_scaling"][scaling_name]["ground_truth"]
        )[-1]
        plt.scatter(ground_truth, val_predictions, alpha=0.5)
        plt.title(f"{scaling_name.capitalize()} Scaling: Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(os.path.join(working_dir, f"{scaling_name}_predictions.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {scaling_name} predictions: {e}")
        plt.close()
