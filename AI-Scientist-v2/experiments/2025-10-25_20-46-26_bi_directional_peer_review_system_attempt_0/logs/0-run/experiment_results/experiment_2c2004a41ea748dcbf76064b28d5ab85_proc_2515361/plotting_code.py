import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for size_key in experiment_data["dataset_size_variation"]:
    try:
        losses = experiment_data["dataset_size_variation"][size_key]["losses"]
        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {size_key}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{size_key}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {size_key}: {e}")
        plt.close()
