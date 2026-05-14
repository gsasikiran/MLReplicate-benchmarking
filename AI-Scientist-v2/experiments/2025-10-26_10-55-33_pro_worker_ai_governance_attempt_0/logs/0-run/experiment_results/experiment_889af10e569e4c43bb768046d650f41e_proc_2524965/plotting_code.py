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

# Training and validation losses plot
for lr, data in experiment_data["varying_learning_rates"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Train Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_plot_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for LR {lr}: {e}")
        plt.close()

# PWIS Plot
try:
    plt.figure()
    for lr, data in experiment_data["varying_learning_rates"].items():
        plt.plot(data["metrics"]["val"], label=f"PWIS (LR: {lr})")
    plt.title("PWIS Metric by Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "pwis_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
