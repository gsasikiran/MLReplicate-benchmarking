import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

datasets = experiment_data["multiple_synthetic_datasets"]

# Plot training and validation metrics
for dataset_name, data in datasets.items():
    epochs = len(data["metrics"]["train"])
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), data["metrics"]["train"], label="Train Metric")
        plt.plot(
            range(1, epochs + 1), data["metrics"]["val"], label="Validation Metric"
        )
        plt.title(f"{dataset_name} - Training and Validation Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics Value")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), data["losses"]["train"], label="Train Loss")
        plt.plot(range(1, epochs + 1), data["losses"]["val"], label="Validation Loss")
        plt.title(f"{dataset_name} - Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.legend()

        plt.savefig(os.path.join(working_dir, f"{dataset_name}_metrics_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {e}")
        plt.close()
