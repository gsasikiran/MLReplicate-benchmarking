import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Train and Validation Losses
try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_lr"]["RQS"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_lr"]["RQS"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "Experiment_RQS_Training_Validation_Losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot for Training and Validation Losses: {e}")
    plt.close()

# Training and Validation Metrics
try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_lr"]["RQS"]["metrics"]["train"],
        label="Train Metric",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_lr"]["RQS"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "Experiment_RQS_Training_Validation_Metrics.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot for Training and Validation Metrics: {e}")
    plt.close()

# Predictions vs Ground Truth
try:
    predictions = np.squeeze(
        np.array(experiment_data["hyperparam_tuning_lr"]["RQS"]["predictions"])
    )
    ground_truth = np.squeeze(
        np.array(experiment_data["hyperparam_tuning_lr"]["RQS"]["ground_truth"])
    )

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.7)
    plt.plot([0, 1], [0, 1], "r--")  # Line of equality
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "Experiment_RQS_Predictions_vs_Ground_Truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot for Predictions vs Ground Truth: {e}")
    plt.close()
