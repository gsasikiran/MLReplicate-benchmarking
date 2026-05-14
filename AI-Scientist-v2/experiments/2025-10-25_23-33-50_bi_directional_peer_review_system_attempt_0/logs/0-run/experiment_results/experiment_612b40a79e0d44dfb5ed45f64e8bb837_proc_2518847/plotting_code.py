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

try:
    plt.figure()
    plt.plot(
        experiment_data["noise_injection"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["noise_injection"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Validation_Loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training and Validation Loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.scatter(
        experiment_data["noise_injection"]["ground_truth"],
        experiment_data["noise_injection"]["predictions"],
        alpha=0.5,
    )
    plt.plot([0, 1], [0, 1], "r--")  # Identity line
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "Predictions_vs_Ground_Truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Predictions vs Ground Truth plot: {e}")
    plt.close()
