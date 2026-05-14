import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training and validation losses for each dropout rate
for dropout_rate, data in experiment_data.items():
    try:
        plt.figure()
        epochs = range(len(data["losses"]["train"]))
        plt.plot(epochs, data["losses"]["train"], label="Training Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {dropout_rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{dropout_rate}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dropout_rate}: {e}")
        plt.close()
