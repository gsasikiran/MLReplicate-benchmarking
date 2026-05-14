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
for dataset_name, results in experiment_data["multiple_synthetic_datasets"].items():
    try:
        plt.figure()
        plt.plot(results["losses"]["train"], label="Training Loss")
        plt.plot(results["losses"]["val"], label="Validation Loss")
        plt.title(f"{dataset_name}: Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

# Plot training metrics if applicable
for dataset_name, results in experiment_data["multiple_synthetic_datasets"].items():
    try:
        plt.figure()
        plt.plot(results["metrics"]["train"], label="Training Metric")
        plt.title(f"{dataset_name}: Training Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_training_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {dataset_name}: {e}")
        plt.close()
