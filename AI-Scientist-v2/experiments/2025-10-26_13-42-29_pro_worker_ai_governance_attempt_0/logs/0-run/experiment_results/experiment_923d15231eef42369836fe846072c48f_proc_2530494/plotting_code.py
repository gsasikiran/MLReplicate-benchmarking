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

for dataset_name, data in experiment_data["Input Feature Correlation Ablation"].items():
    try:
        plt.figure()
        epochs = np.arange(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["val"], label="WWBI")
        plt.title(f"WWBI Metric for {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("WWBI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_wwbi_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating WWBI plot for {dataset_name}: {e}")
        plt.close()
