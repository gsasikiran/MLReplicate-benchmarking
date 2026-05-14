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

for dataset_name in experiment_data["dataset_variation_ablation"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["dataset_variation_ablation"][dataset_name]["metrics"][
                "train"
            ],
            label="Train Accuracy",
        )
        plt.plot(
            experiment_data["dataset_variation_ablation"][dataset_name]["metrics"][
                "val"
            ],
            label="Validation Accuracy",
        )
        plt.title(f"{dataset_name} - Training and Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dataset_name}: {e}")

    try:
        plt.figure()
        plt.plot(
            experiment_data["dataset_variation_ablation"][dataset_name]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["dataset_variation_ablation"][dataset_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name} - Training and Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")

    # Plot Ground Truth vs Predictions for Validation Data
    for epoch in range(0, 50, 10):  # Adjust to plot every 10 epochs
        try:
            plt.figure()
            plt.scatter(
                experiment_data["dataset_variation_ablation"][dataset_name][
                    "ground_truth"
                ][epoch],
                experiment_data["dataset_variation_ablation"][dataset_name][
                    "predictions"
                ][epoch],
                alpha=0.5,
            )
            plt.title(
                f"{dataset_name} - Ground Truth vs Predictions at Epoch {epoch+1}"
            )
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset_name}_epoch_{epoch+1}_gt_vs_pred.png"
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating GT vs Predictions plot for {dataset_name} at epoch {epoch+1}: {e}"
            )
