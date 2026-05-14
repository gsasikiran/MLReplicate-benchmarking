import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for batch_size in experiment_data["batch_size_variation"]:
    try:
        train_losses = experiment_data["batch_size_variation"][batch_size]["losses"][
            "train"
        ]
        val_losses = experiment_data["batch_size_variation"][batch_size]["losses"][
            "val"
        ]
        epochs = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Batch Size {batch_size} - Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{working_dir}/batch_size_{batch_size}_losses.png")
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {batch_size} losses: {e}")
        plt.close()

    try:
        predictions = np.array(
            experiment_data["batch_size_variation"][batch_size]["predictions"]
        )
        ground_truth = np.array(
            experiment_data["batch_size_variation"][batch_size]["ground_truth"]
        )

        plt.figure()
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.title(f"Batch Size {batch_size} - Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(f"{working_dir}/batch_size_{batch_size}_predictions_gt.png")
        plt.close()
    except Exception as e:
        print(
            f"Error creating plot for batch size {batch_size} predictions vs ground truth: {e}"
        )
        plt.close()
