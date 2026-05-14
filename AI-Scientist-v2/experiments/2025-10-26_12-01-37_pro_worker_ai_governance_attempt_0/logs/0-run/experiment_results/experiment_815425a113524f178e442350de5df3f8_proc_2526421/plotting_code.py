import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot loss curves for each dataset
for dataset_name, data in experiment_data["ablation_study"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(data["ground_truth"], data["predictions"], alpha=0.5)
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.title(f"{dataset_name} Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions scatter plot: {e}")
        plt.close()
