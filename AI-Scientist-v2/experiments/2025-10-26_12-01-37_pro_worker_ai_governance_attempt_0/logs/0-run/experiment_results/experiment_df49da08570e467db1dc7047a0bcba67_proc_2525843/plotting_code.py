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

# Plotting training and validation loss curves
for lr in experiment_data:
    try:
        plt.figure()
        plt.plot(experiment_data[lr]["losses"]["train"], label="Training Loss")
        plt.plot(experiment_data[lr]["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curve for Learning Rate {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"Loss_Curve_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {lr}: {e}")
        plt.close()

# Plotting predictions vs ground truth for the last learning rate
last_lr = list(experiment_data.keys())[-1]
try:
    plt.figure()
    plt.scatter(
        experiment_data[last_lr]["ground_truth"],
        experiment_data[last_lr]["predictions"],
        alpha=0.5,
    )
    plt.plot([0, 1], [0, 1], "r--")  # Ideal line
    plt.title(f"Ground Truth vs Predictions for Learning Rate {last_lr}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, f"Ground_Truth_vs_Predictions_{last_lr}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot for {last_lr}: {e}")
    plt.close()
