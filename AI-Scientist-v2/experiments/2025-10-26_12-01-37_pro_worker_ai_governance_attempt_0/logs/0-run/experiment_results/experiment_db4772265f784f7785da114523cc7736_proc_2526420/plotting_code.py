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

try:
    architectures = experiment_data["model_architecture_variation"]
    for key in architectures:
        plt.figure()
        plt.plot(architectures[key]["losses"]["train"], label="Training Loss")
        plt.plot(architectures[key]["losses"]["val"], label="Validation Loss")
        plt.title(f"{key} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plots: {e}")
    plt.close()

try:
    for key in architectures:
        plt.figure()
        plt.plot(architectures[key]["metrics"]["val"], label="Validation Metric (EIS)")
        plt.title(f"{key} Validation Metric Curves")
        plt.xlabel("Epochs")
        plt.ylabel("EIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_validation_metric.png"))
        plt.close()
except Exception as e:
    print(f"Error creating validation metric plots: {e}")
    plt.close()
