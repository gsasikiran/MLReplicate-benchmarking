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

num_epochs_list = [100, 200, 300]

# Plotting loss curves
for num_epochs in num_epochs_list:
    try:
        train_losses = experiment_data["hyperparam_tuning_num_epochs"][
            f"epochs_{num_epochs}"
        ]["losses"]["train"]
        val_losses = experiment_data["hyperparam_tuning_num_epochs"][
            f"epochs_{num_epochs}"
        ]["losses"]["val"]

        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for Epochs {num_epochs}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{working_dir}/loss_curves_epochs_{num_epochs}.png")
        plt.close()
    except Exception as e:
        print(f"Error creating plot for epochs {num_epochs}: {e}")
        plt.close()
