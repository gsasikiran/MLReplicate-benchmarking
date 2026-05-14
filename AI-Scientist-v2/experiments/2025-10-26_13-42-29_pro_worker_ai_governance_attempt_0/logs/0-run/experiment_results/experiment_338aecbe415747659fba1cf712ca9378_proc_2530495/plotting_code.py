import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    for batch_size in experiment_data["variable_batch_size"]:
        metric_data = experiment_data["variable_batch_size"][batch_size]["losses"]
        epochs = range(len(metric_data["train"]))

        # Plot losses
        plt.figure()
        plt.plot(epochs, metric_data["train"], label="Training Loss")
        plt.plot(epochs, metric_data["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_{batch_size}.png"))
        plt.close()

        # Plot predictions versus ground truth for the best model
        if (
            batch_size == "batch_size_32"
        ):  # Assuming this size gives the best performance
            predictions = experiment_data["variable_batch_size"][batch_size][
                "predictions"
            ]
            ground_truth = experiment_data["variable_batch_size"][batch_size][
                "ground_truth"
            ]
            plt.figure()
            plt.scatter(ground_truth, predictions, alpha=0.7)
            plt.plot([0, 1], [0, 1], "r--")  # Line of perfect prediction
            plt.title(f"Predictions vs Ground Truth for {batch_size}")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.savefig(
                os.path.join(
                    working_dir, f"predictions_vs_ground_truth_{batch_size}.png"
                )
            )
            plt.close()
except Exception as e:
    print(f"Error while processing experiment data: {e}")
