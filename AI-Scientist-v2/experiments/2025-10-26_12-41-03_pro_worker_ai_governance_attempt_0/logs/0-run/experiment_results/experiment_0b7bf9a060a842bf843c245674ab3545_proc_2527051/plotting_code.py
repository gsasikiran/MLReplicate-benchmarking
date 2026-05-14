import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    plt.plot(experiment_data["synthetic_data"]["losses"]["train"], label="Train Loss")
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.scatter(
        experiment_data["synthetic_data"]["ground_truth"],
        experiment_data["synthetic_data"]["predictions"],
        alpha=0.5,
    )
    plt.title("Ground Truth vs Predictions")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.axline((0, 0), slope=1, color="red", linestyle="--")
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_ground_truth_vs_predictions.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
