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

# Training and validation losses plot for each optimizer
for optimizer in experiment_data["hyperparam_tuning_optimizers"]:
    try:
        train_losses = experiment_data["hyperparam_tuning_optimizers"][optimizer][
            "losses"
        ]["train"]
        val_losses = experiment_data["hyperparam_tuning_optimizers"][optimizer][
            "losses"
        ]["val"]
        epochs = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{optimizer} Loss Plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {optimizer}: {e}")
        plt.close()
