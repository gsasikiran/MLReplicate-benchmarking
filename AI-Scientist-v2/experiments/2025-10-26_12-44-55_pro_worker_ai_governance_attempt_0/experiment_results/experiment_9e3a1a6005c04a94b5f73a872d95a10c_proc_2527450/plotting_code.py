import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    train_losses = experiment_data["hyperparam_tuning_dropout"][
        "synthetic_worker_data"
    ]["losses"]["train"]
    val_losses = experiment_data["hyperparam_tuning_dropout"]["synthetic_worker_data"][
        "losses"
    ]["val"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot predicted vs ground truth values
try:
    predictions = experiment_data["hyperparam_tuning_dropout"]["synthetic_worker_data"][
        "predictions"
    ]
    ground_truth = experiment_data["hyperparam_tuning_dropout"][
        "synthetic_worker_data"
    ]["ground_truth"]
    plt.figure()
    plt.scatter(ground_truth, predictions)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        "r--",
    )  # Line y=x
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
