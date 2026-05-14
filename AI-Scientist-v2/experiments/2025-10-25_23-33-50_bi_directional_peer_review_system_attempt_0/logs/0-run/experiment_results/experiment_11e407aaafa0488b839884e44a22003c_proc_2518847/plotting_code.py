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

# Plot for training and validation losses without gradient clipping
try:
    plt.figure()
    plt.plot(
        experiment_data["gradient_clipping"]["without_clipping"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["gradient_clipping"]["without_clipping"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss - Without Gradient Clipping")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "train_val_loss_no_clipping.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot without clipping: {e}")
    plt.close()

# Plot for training and validation losses with gradient clipping
try:
    plt.figure()
    plt.plot(
        experiment_data["gradient_clipping"]["with_clipping"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["gradient_clipping"]["with_clipping"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss - With Gradient Clipping")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "train_val_loss_with_clipping.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot with clipping: {e}")
    plt.close()
