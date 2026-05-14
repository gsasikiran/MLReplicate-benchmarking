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

for batch_size in experiment_data["batch_size_tuning"].keys():
    try:
        plt.figure()
        train_losses = experiment_data["batch_size_tuning"][batch_size]["losses"][
            "train"
        ]
        val_losses = experiment_data["batch_size_tuning"][batch_size]["losses"]["val"]
        epochs = range(len(train_losses))

        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for Batch Size {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"loss_curves_batch_size_{batch_size}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {batch_size}: {e}")
        plt.close()
