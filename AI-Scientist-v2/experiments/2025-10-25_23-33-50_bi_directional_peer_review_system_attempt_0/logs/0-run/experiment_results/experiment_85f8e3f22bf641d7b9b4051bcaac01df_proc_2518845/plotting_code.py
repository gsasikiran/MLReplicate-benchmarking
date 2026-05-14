import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for dataset_name, results in experiment_data["multi_dataset_eval"].items():
    try:
        plt.figure()
        epochs = range(1, len(results["losses"]["train"]) + 1)
        plt.plot(epochs, results["losses"]["train"], label="Train Loss")
        plt.plot(epochs, results["losses"]["val"], label="Validation Loss")
        plt.title(f"{dataset_name} Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        epochs = range(1, len(results["metrics"]["train"]) + 1)
        plt.plot(epochs, results["metrics"]["train"], label="Train RQI")
        plt.plot(epochs, results["metrics"]["val"], label="Validation RQI")
        plt.title(f"{dataset_name} Training and Validation RQI")
        plt.xlabel("Epochs")
        plt.ylabel("RQI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_rqi_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RQI plot for {dataset_name}: {e}")
        plt.close()
