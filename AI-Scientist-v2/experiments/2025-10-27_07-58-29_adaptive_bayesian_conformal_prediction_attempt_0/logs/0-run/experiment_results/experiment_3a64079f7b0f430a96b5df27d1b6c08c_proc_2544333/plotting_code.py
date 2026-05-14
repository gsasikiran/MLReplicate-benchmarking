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

# Plot training and validation losses
for key in experiment_data.keys():
    try:
        plt.figure()
        plt.plot(experiment_data[key]["losses"]["train"], label="Train Loss")
        plt.plot(experiment_data[key]["losses"]["val"], label="Validation Loss")
        plt.title(f"{key}: Loss Plot")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {key} loss plot: {e}")
        plt.close()

# Plot metrics
for key in experiment_data.keys():
    try:
        plt.figure()
        plt.plot(experiment_data[key]["metrics"]["val"], label="Validation ACIW")
        plt.title(f"{key}: Validation ACIW Plot")
        plt.xlabel("Epochs")
        plt.ylabel("ACIW")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_aciw_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {key} ACIW plot: {e}")
        plt.close()
