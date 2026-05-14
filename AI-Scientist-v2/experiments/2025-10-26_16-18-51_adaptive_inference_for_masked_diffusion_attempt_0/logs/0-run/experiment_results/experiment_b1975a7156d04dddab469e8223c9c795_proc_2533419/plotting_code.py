import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for optimizer in experiment_data["use_of_different_optimizers"]:
    losses = experiment_data["use_of_different_optimizers"][optimizer]["losses"]

    try:
        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"{optimizer} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {optimizer}: {e}")
        plt.close()
