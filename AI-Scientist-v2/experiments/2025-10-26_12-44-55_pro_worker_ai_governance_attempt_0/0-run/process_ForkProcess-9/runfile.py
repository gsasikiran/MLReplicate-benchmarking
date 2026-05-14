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

# Plot training and validation losses
try:
    losses = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_worker_data"
    ]["losses"]
    epochs = range(len(losses["train"]))

    plt.figure()
    plt.plot(epochs, losses["train"], label="Training Loss")
    plt.plot(epochs, losses["val"], label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_training_validation_losses.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot predictions vs ground truth
try:
    predictions = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_worker_data"
    ]["predictions"]
    ground_truth = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_worker_data"
    ]["ground_truth"]

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        color="red",
        linestyle="--",
    )
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth WIS")
    plt.ylabel("Predicted WIS")
    plt.axis("equal")
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
