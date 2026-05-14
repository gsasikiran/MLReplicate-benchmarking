import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for scale_type in experiment_data["feature_scaling"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["feature_scaling"][scale_type]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["feature_scaling"][scale_type]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{scale_type} Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{scale_type}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {scale_type} loss curve: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["feature_scaling"][scale_type]["metrics"]["val"],
            label="Economic Impact Score",
        )
        plt.title(f"{scale_type} Economic Impact Score Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("EIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{scale_type}_EIS_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {scale_type} EIS curve: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(
            experiment_data["feature_scaling"][scale_type]["ground_truth"],
            experiment_data["feature_scaling"][scale_type]["predictions"],
        )
        plt.title(f"{scale_type}: Ground Truth vs Predictions")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(os.path.join(working_dir, f"{scale_type}_GT_vs_Predictions.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {scale_type} GT vs Predictions scatter plot: {e}")
        plt.close()
