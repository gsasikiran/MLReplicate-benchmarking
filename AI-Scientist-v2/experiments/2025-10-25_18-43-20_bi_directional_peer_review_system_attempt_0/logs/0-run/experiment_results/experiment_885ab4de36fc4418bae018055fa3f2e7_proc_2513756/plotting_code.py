import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation loss curves
for i in range(1, 4):
    try:
        dataset_name = f"dataset_{i}"
        train_losses = experiment_data["dataset_diversity_ablation"][dataset_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["dataset_diversity_ablation"][dataset_name][
            "losses"
        ]["val"]

        plt.figure()
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="orange")
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {e}")
        plt.close()
