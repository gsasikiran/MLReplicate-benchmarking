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

for randomness_level in range(1, 4):
    dataset_name = f"dataset_randomness_{randomness_level}"
    try:
        plt.figure()
        train_losses = experiment_data["multi_dataset_evaluation"][dataset_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["multi_dataset_evaluation"][dataset_name][
            "losses"
        ]["val"]
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {e}")
        plt.close()
