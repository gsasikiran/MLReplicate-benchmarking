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

try:
    plt.figure()
    plt.plot(
        experiment_data["batch_normalization"]["sudoku"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["batch_normalization"]["sudoku"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Sudoku Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()
