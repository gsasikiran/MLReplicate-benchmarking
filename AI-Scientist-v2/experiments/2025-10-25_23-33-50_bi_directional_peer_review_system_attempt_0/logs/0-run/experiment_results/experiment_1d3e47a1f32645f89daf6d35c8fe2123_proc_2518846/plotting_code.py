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

for distribution in ["uniform", "normal", "skewed"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][distribution]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["ablation_study"][distribution]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{distribution.capitalize()} Distribution - Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{distribution}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {distribution}: {e}")
        plt.close()

    try:
        plt.figure()
        rqi = [
            1 - loss
            for loss in experiment_data["ablation_study"][distribution]["losses"]["val"]
        ]
        plt.plot(rqi, label="RQI")
        plt.title(f"{distribution.capitalize()} Distribution - RQI Curve")
        plt.xlabel("Epochs")
        plt.ylabel("RQI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{distribution}_rqi_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RQI plot for {distribution}: {e}")
        plt.close()

    # Plot predictions vs ground truth
    try:
        plt.figure()
        plt.scatter(
            experiment_data["ablation_study"][distribution]["ground_truth"],
            experiment_data["ablation_study"][distribution]["predictions"],
            alpha=0.5,
        )
        plt.title(
            f"{distribution.capitalize()} Distribution - Predictions vs Ground Truth"
        )
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"{distribution}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {distribution}: {e}")
        plt.close()
