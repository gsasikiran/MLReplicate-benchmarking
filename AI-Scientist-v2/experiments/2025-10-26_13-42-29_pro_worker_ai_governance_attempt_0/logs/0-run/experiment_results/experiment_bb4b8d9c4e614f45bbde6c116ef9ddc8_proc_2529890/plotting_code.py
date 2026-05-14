import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation loss for different weight decays
for wd in experiment_data["weight_decay_tuning"]:
    try:
        epochs = range(
            1, len(experiment_data["weight_decay_tuning"][wd]["losses"]["train"]) + 1
        )
        plt.figure()
        plt.plot(
            epochs,
            experiment_data["weight_decay_tuning"][wd]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            epochs,
            experiment_data["weight_decay_tuning"][wd]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for Weight Decay: {wd}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_wd_{wd}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for weight decay {wd}: {e}")

# Plot predictions vs ground truth for each weight decay
for wd in experiment_data["weight_decay_tuning"]:
    try:
        plt.figure()
        plt.scatter(
            experiment_data["weight_decay_tuning"][wd]["ground_truth"],
            experiment_data["weight_decay_tuning"][wd]["predictions"],
            alpha=0.5,
        )
        plt.plot([0, 1], [0, 1], "r--")  # Diagonal line
        plt.title(f"Predictions vs Ground Truth for Weight Decay: {wd}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"predictions_vs_ground_truth_wd_{wd}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for weight decay {wd}: {e}")
