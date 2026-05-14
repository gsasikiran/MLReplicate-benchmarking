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

for feature_name, data in experiment_data["feature_importance"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Train Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {feature_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{feature_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {feature_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["val"], label="PWIS (Validation Metric)")
        plt.title(f"PWIS Metric for {feature_name}")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"PWIS_{feature_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PWIS plot for {feature_name}: {e}")
        plt.close()
