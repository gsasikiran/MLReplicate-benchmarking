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

for dataset_name in experiment_data["multi_dataset_variation"]:
    data = experiment_data["multi_dataset_variation"][dataset_name]
    epochs = range(len(data["metrics"]["train"]))

    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["train"], label="Training Loss")
        plt.title(f"{dataset_name.capitalize()} Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(epochs, data["metrics"]["CIS"], label="CIS Metric", color="orange")
        plt.title(f"{dataset_name.capitalize()} Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("CIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_CIS_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CIS metric plot for {dataset_name}: {e}")
        plt.close()
