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

for method in ["uniform", "xavier", "he"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["weight_initialization_ablation"][method]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["weight_initialization_ablation"][method]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for {method} Initialization")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{method}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {method}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(
            experiment_data["weight_initialization_ablation"][method]["ground_truth"],
            experiment_data["weight_initialization_ablation"][method]["predictions"][
                -1
            ],
            label="Predictions vs Ground Truth",
        )
        plt.plot([0, 3], [0, 3], "r--")  # Line y=x for reference
        plt.title(f"Predicted vs Ground Truth for {method} Initialization")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"predictions_vs_ground_truth_{method}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {method}: {e}")
        plt.close()
