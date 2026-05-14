import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for dataset_type in ["Uniform", "Normal", "Exponential"]:
    try:
        train_losses = experiment_data["multiple_synthetic_datasets"][dataset_type][
            "losses"
        ]["train"]
        val_losses = experiment_data["multiple_synthetic_datasets"][dataset_type][
            "losses"
        ]["val"]

        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for {dataset_type} Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_type}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_type} dataset: {e}")
        plt.close()
