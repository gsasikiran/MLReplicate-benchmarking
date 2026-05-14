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

for dataset_name, data in experiment_data["multiple_synthetic_datasets"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.title(f"{dataset_name} - Training Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="CODS")
        plt.title(f"{dataset_name} - CODS over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("CODS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_cods.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CODS plot: {e}")
        plt.close()
