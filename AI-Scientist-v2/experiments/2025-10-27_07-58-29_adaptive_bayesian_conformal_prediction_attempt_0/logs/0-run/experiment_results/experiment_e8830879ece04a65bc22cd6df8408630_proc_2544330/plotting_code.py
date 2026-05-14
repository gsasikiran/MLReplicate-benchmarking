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

for noise_name in experiment_data["multiple_synthetic_datasets"]:
    try:
        plt.figure()
        epochs = np.arange(1, 101)
        train_losses = experiment_data["multiple_synthetic_datasets"][noise_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["multiple_synthetic_datasets"][noise_name][
            "losses"
        ]["val"]
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for {noise_name.capitalize()} Noise Level")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{noise_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {noise_name}: {e}")
        plt.close()
