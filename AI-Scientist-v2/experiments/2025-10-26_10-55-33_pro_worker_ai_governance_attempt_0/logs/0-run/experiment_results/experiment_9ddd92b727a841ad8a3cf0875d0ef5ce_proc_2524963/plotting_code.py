import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses for each noise level
for noise_level, data in experiment_data["varying_input_noise"].items():
    try:
        plt.figure()
        epochs = range(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {noise_level}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{noise_level}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot for {noise_level}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["val"], label="PWIS Metric")
        plt.title(f"PWIS Metric for {noise_level}")
        plt.xlabel("Epoch")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"PWIS_metric_{noise_level}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PWIS plot for {noise_level}: {e}")
        plt.close()
