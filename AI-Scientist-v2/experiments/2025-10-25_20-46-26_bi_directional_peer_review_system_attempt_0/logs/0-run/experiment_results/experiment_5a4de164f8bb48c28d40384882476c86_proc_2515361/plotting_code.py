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

activation_names = list(experiment_data["activation_depth_variation"].keys())

for act_name in activation_names:
    try:
        losses = experiment_data["activation_depth_variation"][act_name]["losses"]
        epochs = range(1, len(losses["train"]) + 1)

        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Validation Loss")
        plt.title(f"{act_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name} loss curves: {e}")
        plt.close()

    try:
        predictions = np.concatenate(
            experiment_data["activation_depth_variation"][act_name]["predictions"]
        )
        ground_truth = np.concatenate(
            experiment_data["activation_depth_variation"][act_name]["ground_truth"]
        )

        plt.figure()
        plt.scatter(ground_truth, predictions, alpha=0.7)
        plt.title(f"{act_name} Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(os.path.join(working_dir, f"{act_name}_predictions_vs_gt.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name} predictions vs ground truth: {e}")
        plt.close()
