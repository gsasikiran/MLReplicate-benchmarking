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

for dataset_name in experiment_data["multiple_synthetic_datasets"]:
    try:
        train_losses = experiment_data["multiple_synthetic_datasets"][dataset_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["multiple_synthetic_datasets"][dataset_name][
            "losses"
        ]["val"]
        epochs = [5, 10, 15, 20]

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for {dataset_name} Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()
