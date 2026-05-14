import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
learning_rates = experiment_data["learning_rate_ablation"].keys()
for lr in learning_rates:
    try:
        plt.figure()
        train_losses = experiment_data["learning_rate_ablation"][lr]["losses"]["train"]
        val_losses = experiment_data["learning_rate_ablation"][lr]["losses"]["val"]
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {lr}: {e}")
        plt.close()

# Plot predictions vs ground truth
for lr in learning_rates:
    try:
        plt.figure()
        predictions = experiment_data["learning_rate_ablation"][lr]["predictions"]
        ground_truth = experiment_data["learning_rate_ablation"][lr]["ground_truth"]
        plt.scatter(ground_truth, predictions)
        plt.title(f"Predictions vs Ground Truth for Learning Rate: {lr}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.plot([0, 1], [0, 1], "r--")  # diagonal line for reference
        plt.savefig(
            os.path.join(working_dir, f"predictions_vs_ground_truth_lr_{lr}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {lr}: {e}")
        plt.close()
