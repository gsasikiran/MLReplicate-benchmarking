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

for dataset_index in range(1, 4):
    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][f"dataset_{dataset_index}"][
                "losses"
            ]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][f"dataset_{dataset_index}"][
                "losses"
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"Dataset {dataset_index} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"dataset_{dataset_index}_loss_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for dataset_{dataset_index}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(
            experiment_data["multiple_synthetic_datasets"][f"dataset_{dataset_index}"][
                "ground_truth"
            ],
            experiment_data["multiple_synthetic_datasets"][f"dataset_{dataset_index}"][
                "predictions"
            ],
            alpha=0.5,
        )
        plt.plot([0, 1], [0, 1], "r--")  # Reference line
        plt.title(f"Dataset {dataset_index} Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"dataset_{dataset_index}_predictions.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for dataset_{dataset_index}: {e}")
        plt.close()
