import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot Training Loss
try:
    plt.figure()
    plt.plot(
        experiment_data["pwis_experiment"]["losses"]["train"], label="Training Loss"
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "pwis_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot Predictions vs Ground Truth
try:
    plt.figure()
    plt.scatter(
        experiment_data["pwis_experiment"]["ground_truth"],
        experiment_data["pwis_experiment"]["predictions"],
        alpha=0.7,
    )
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth PWIS")
    plt.ylabel("Predicted PWIS")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # identity line
    plt.savefig(os.path.join(working_dir, "pwis_predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
