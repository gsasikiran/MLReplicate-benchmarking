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

for rate in experiment_data["dropout_tuning"]:
    try:
        plt.figure()
        train_losses = experiment_data["dropout_tuning"][rate]["losses"]["train"]
        val_losses = experiment_data["dropout_tuning"][rate]["losses"]["val"]
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for Dropout Rate {rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{working_dir}/loss_curves_dropout_{rate}.png")
        plt.close()
    except Exception as e:
        print(f"Error creating plot for dropout {rate}: {e}")
        plt.close()
