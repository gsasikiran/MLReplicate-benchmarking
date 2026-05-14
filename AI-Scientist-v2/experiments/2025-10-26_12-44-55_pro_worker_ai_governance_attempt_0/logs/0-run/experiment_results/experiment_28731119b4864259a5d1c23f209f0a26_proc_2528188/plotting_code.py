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

for dataset_name in experiment_data["ablation_study"].keys():
    try:
        train_losses = experiment_data["ablation_study"][dataset_name]["losses"][
            "train"
        ]
        val_losses = experiment_data["ablation_study"][dataset_name]["losses"]["val"]
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        predictions = experiment_data["ablation_study"][dataset_name]["predictions"]
        ground_truth = experiment_data["ablation_study"][dataset_name]["ground_truth"]
        plt.figure()
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.title(f"Predictions vs Ground Truth for {dataset_name}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {dataset_name}: {e}")
        plt.close()
