import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    epochs = len(experiment_data["weight_decay_tuning"]["dataset"]["metrics"]["train"])

    # Plot training and validation metrics
    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        experiment_data["weight_decay_tuning"]["dataset"]["metrics"]["train"],
        label="Training Metric",
    )
    plt.plot(
        range(1, epochs + 1),
        experiment_data["weight_decay_tuning"]["dataset"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "experiment_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

try:
    # Plot training and validation losses
    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        experiment_data["weight_decay_tuning"]["dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        range(1, epochs + 1),
        experiment_data["weight_decay_tuning"]["dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "experiment_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot predicted vs ground truth for the last epoch
    y_val_predictions = experiment_data["weight_decay_tuning"]["dataset"][
        "predictions"
    ][0][-1]
    y_val_truth = experiment_data["weight_decay_tuning"]["dataset"]["ground_truth"][0][
        -1
    ]

    plt.figure()
    plt.scatter(y_val_truth, y_val_predictions)
    plt.plot([0, 1], [0, 1], "r--")  # Ideal prediction line
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "predictions_vs_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
