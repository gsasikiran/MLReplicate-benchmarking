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

datasets = ["default", "alternate1", "alternate2"]

for dataset in datasets:
    try:
        # Plotting training and validation losses
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][dataset]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["multi_dataset_evaluation"][dataset]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{dataset.capitalize()} Dataset Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset} losses: {e}")
        plt.close()

    try:
        # Plotting metrics
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][dataset]["metrics"]["train"],
            label="Training Metric",
        )
        plt.title(f"{dataset.capitalize()} Dataset Training Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset} metrics: {e}")
        plt.close()
