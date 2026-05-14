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

for noise_key, noise_data in experiment_data["noise_level_impact"].items():
    # Plot training and validation losses
    try:
        plt.figure()
        plt.plot(noise_data["losses"]["train"], label="Training Loss")
        plt.plot(noise_data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {noise_key}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{noise_key}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {noise_key}: {e}")
        plt.close()

    # Plot predictions vs ground truth at the last 5 epochs
    try:
        plt.figure()
        for i, (pred, gt) in enumerate(
            zip(noise_data["predictions"], noise_data["ground_truth"])
        ):
            if i % 20 == 0:  # Plot every 20th prediction
                plt.scatter(gt, pred, label=f"Epoch {i+1}")
        plt.title(f"{noise_key}: Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{noise_key}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {noise_key}: {e}")
        plt.close()
