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

for dataset_name, data in experiment_data.items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{dataset_name}: Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["val"], label="Reliability Measure")
        plt.title(f"{dataset_name}: Validation Reliability Measure")
        plt.xlabel("Epochs")
        plt.ylabel("Reliability Measure")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_reliability_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating reliability plot for {dataset_name}: {e}")
        plt.close()
