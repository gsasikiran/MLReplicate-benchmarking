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

for missing_rate, data in experiment_data["multiple_synthetic_datasets"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Train Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves - {missing_rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{missing_rate}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {missing_rate}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="Train Metric")
        plt.plot([m[0] for m in data["metrics"]["val"]], label="Validation Metric")
        plt.title(f"Metric Curves - {missing_rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{missing_rate}_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {missing_rate}: {e}")
        plt.close()
