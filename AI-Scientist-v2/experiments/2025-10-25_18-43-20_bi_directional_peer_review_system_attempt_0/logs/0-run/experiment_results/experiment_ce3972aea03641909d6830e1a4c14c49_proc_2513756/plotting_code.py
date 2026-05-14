import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for dataset_type in ["Uniform", "Normal", "Exponential"]:
    try:
        epochs = list(
            range(
                len(
                    experiment_data["multiple_synthetic_datasets"][dataset_type][
                        "losses"
                    ]["train"]
                )
            )
        )
        train_losses = experiment_data["multiple_synthetic_datasets"][dataset_type][
            "losses"
        ]["train"]
        val_losses = experiment_data["multiple_synthetic_datasets"][dataset_type][
            "losses"
        ]["val"]

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{dataset_type} Dataset Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_type}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_type} dataset: {e}")
        plt.close()
