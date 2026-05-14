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
    train_losses = experiment_data["weight_decay_tuning"]["synthetic_worker_data"][
        "losses"
    ]["train"]
    val_losses = experiment_data["weight_decay_tuning"]["synthetic_worker_data"][
        "losses"
    ]["val"]

    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curves for Synthetic Worker Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_synthetic_worker_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot ground truth vs predictions
try:
    predictions = np.array(
        experiment_data["weight_decay_tuning"]["synthetic_worker_data"]["predictions"]
    )
    ground_truth = np.array(
        experiment_data["weight_decay_tuning"]["synthetic_worker_data"]["ground_truth"]
    )

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [ground_truth.min(), ground_truth.max()],
        [ground_truth.min(), ground_truth.max()],
        color="red",
        linestyle="dashed",
    )
    plt.title("Ground Truth vs Predictions for Synthetic Worker Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.savefig(
        os.path.join(
            working_dir, "ground_truth_vs_predictions_synthetic_worker_data.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
