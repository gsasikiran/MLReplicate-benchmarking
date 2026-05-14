import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for complexity in ["simple", "moderate", "complex"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_robustness"][complexity]["losses"]["train"],
            label="Training Loss",
        )
        plt.title(f"{complexity} Dataset: Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{complexity}_dataset_training_loss.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {complexity} dataset training loss: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_robustness"][complexity]["metrics"]["train"],
            label="UES Metric",
        )
        plt.title(f"{complexity} Dataset: UES Metric Curve")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{complexity}_dataset_ues_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {complexity} dataset UES metric: {e}")
        plt.close()
