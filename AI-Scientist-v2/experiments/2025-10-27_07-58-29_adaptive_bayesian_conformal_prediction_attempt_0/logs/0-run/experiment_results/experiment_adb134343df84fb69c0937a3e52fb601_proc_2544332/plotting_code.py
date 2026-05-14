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

for key in experiment_data.keys():
    try:
        plt.figure()
        plt.plot(
            experiment_data[key]["synthetic_data"]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data[key]["synthetic_data"]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{key}: Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {key} loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(
            experiment_data[key]["synthetic_data"]["ground_truth"][-1],
            experiment_data[key]["synthetic_data"]["predictions"][-1],
        )
        plt.title(f"{key}: Ground Truth vs Predictions")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.plot(
            [
                min(experiment_data[key]["synthetic_data"]["ground_truth"][-1]),
                max(experiment_data[key]["synthetic_data"]["ground_truth"][-1]),
            ],
            [
                min(experiment_data[key]["synthetic_data"]["ground_truth"][-1]),
                max(experiment_data[key]["synthetic_data"]["ground_truth"][-1]),
            ],
            color="red",
            linestyle="--",
        )  # diagonal for comparison
        plt.savefig(os.path.join(working_dir, f"{key}_ground_truth_vs_predictions.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {key} ground truth vs predictions plot: {e}")
        plt.close()
