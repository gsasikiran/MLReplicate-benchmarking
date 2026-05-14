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

for momentum in [0.0, 0.5, 0.9]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["momentum_tuning"]["sudoku"]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["momentum_tuning"]["sudoku"]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Training and Validation Loss for Momentum = {momentum}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"sudoku_loss_momentum_{momentum}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for momentum {momentum}: {e}")
        plt.close()
