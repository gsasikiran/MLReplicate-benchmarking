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

for act_name in experiment_data["activation_merging"]:
    try:
        losses = experiment_data["activation_merging"][act_name]["losses"]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses["train"], label="Train Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"{act_name} - Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {act_name}: {e}")
        plt.close()

    try:
        rqs = experiment_data["activation_merging"][act_name]["rqs"]

        plt.figure(figsize=(6, 5))
        plt.plot(rqs, label="Review Quality Score")
        plt.title(f"{act_name} - Review Quality Scores")
        plt.xlabel("Epochs")
        plt.ylabel("RQS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_rqs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RQS plot for {act_name}: {e}")
        plt.close()
