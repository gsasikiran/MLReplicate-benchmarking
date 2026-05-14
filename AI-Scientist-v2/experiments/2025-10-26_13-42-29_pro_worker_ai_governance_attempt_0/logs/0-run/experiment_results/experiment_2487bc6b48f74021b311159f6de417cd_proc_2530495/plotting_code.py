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

for dataset_name, data in experiment_data["multi_dataset_ablation"].items():
    for wd_key, wd_data in data["weight_decay_tuning"].items():
        try:
            plt.figure()
            plt.plot(wd_data["losses"]["train"], label="Training Loss")
            plt.plot(wd_data["losses"]["val"], label="Validation Loss")
            plt.title(f"Loss Curves for {dataset_name} with {wd_key}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_{wd_key}_loss_curve.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dataset_name} and {wd_key}: {e}")
            plt.close()
