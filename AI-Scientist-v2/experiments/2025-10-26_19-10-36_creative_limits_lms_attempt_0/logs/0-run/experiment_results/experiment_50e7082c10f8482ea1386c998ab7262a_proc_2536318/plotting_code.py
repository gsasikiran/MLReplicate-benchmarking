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

for batch_size in experiment_data["batch_size_tuning"]:
    try:
        losses = experiment_data["batch_size_tuning"][batch_size]["losses"]["train"]
        epochs = range(1, len(losses) + 1)
        plt.figure()
        plt.plot(epochs, losses, label="Training Loss")
        plt.title(f"Training Loss (Batch Size: {batch_size})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_batch_{batch_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for training loss (batch size {batch_size}): {e}")
        plt.close()

    try:
        cods = experiment_data["batch_size_tuning"][batch_size]["metrics"]["train"]
        plt.figure()
        plt.plot(epochs, cods, label="CODS", color="orange")
        plt.title(f"CODS (Batch Size: {batch_size})")
        plt.xlabel("Epochs")
        plt.ylabel("CODS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"cods_batch_{batch_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for CODS (batch size {batch_size}): {e}")
        plt.close()
