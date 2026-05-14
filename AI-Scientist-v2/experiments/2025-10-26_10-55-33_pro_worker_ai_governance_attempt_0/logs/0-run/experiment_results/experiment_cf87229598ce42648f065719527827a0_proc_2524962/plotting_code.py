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

for model_name, data in experiment_data["different_model_architectures"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{model_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {model_name} loss curves: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["val"], label="PWIS (Validation Metric)")
        plt.title(f"{model_name} Validation Metric Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_name}_validation_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {model_name} validation metric: {e}")
        plt.close()

    if data["predictions"]:
        try:
            plt.figure()
            plt.scatter(data["ground_truth"][0], data["predictions"][0], alpha=0.5)
            plt.title(f"{model_name} Predictions vs Ground Truth")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.axis("equal")
            plt.grid()
            plt.savefig(
                os.path.join(
                    working_dir, f"{model_name}_predictions_vs_ground_truth.png"
                )
            )
            plt.close()
        except Exception as e:
            print(f"Error creating scatter plot for {model_name}: {e}")
            plt.close()
