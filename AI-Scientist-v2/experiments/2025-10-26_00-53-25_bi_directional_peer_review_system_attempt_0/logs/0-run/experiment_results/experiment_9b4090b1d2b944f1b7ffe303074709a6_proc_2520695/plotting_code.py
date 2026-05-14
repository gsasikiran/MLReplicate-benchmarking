import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Iterate over noise levels and create plots
noise_levels = list(experiment_data["noise_robustness"].keys())
for noise in noise_levels:
    metrics = experiment_data["noise_robustness"][noise]["metrics"]
    losses = experiment_data["noise_robustness"][noise]["losses"]

    try:
        plt.figure()
        plt.plot(metrics["train"], label="Train Accuracy")
        plt.plot(metrics["val"], label="Validation Accuracy")
        plt.title(f"Noise Level: {noise} - Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{noise}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {noise}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(losses["train"], label="Train Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Noise Level: {noise} - Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{noise}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {noise}: {e}")
        plt.close()
