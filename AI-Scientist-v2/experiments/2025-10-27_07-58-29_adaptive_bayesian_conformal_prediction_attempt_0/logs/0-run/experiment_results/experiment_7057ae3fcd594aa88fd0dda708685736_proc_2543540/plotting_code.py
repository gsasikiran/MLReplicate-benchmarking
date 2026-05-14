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
    metrics = experiment_data["hyperparam_tuning_activation_function"][
        "synthetic_data"
    ]["losses"]
    epochs = range(len(metrics["train"]))

    plt.figure()
    plt.plot(epochs, metrics["train"], label="Training Loss")
    plt.plot(epochs, metrics["val"], label="Validation Loss")
    plt.title("Loss Curves - Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    predictions = experiment_data["hyperparam_tuning_activation_function"][
        "synthetic_data"
    ]["predictions"]
    ground_truth = experiment_data["hyperparam_tuning_activation_function"][
        "synthetic_data"
    ]["ground_truth"][
        0
    ]  # Use the first epoch ground truth
    plt.figure()
    plt.scatter(ground_truth, predictions[0], label="Predictions", alpha=0.5)
    plt.plot(
        ground_truth, ground_truth, color="red", label="Ground Truth", linestyle="--"
    )
    plt.title("Predictions vs Ground Truth - Synthetic Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "predictions_vs_ground_truth_synthetic_data.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
