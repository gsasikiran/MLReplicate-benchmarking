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

for dataset_name in ["easy", "medium", "hard"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["dataset_variation"][dataset_name]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["dataset_variation"][dataset_name]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name.capitalize()} Dataset Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} loss: {e}")
        plt.close()

    try:
        plt.figure()
        predictions = experiment_data["dataset_variation"][dataset_name]["predictions"][
            0
        ]
        ground_truth = experiment_data["dataset_variation"][dataset_name][
            "ground_truth"
        ][0]
        plt.scatter(
            range(len(ground_truth)), ground_truth, label="Ground Truth", alpha=0.5
        )
        plt.scatter(
            range(len(predictions)), predictions, label="Predictions", alpha=0.5
        )
        plt.title(f"{dataset_name.capitalize()} Dataset Predictions vs Ground Truth")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot for {dataset_name}: {e}")
        plt.close()
