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

for dataset_name in experiment_data["multiple_synthetic_datasets"]:
    try:
        losses = experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"]

        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"{dataset_name.capitalize()} Dataset Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {e}")
        plt.close()
