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
    epochs = range(
        1, len(experiment_data["batch_size_ablation"]["RQS"]["losses"]["train"]) + 1
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_ablation"]["RQS"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_ablation"]["RQS"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQS_training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training and Validation Loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["batch_size_ablation"]["RQS"]["metrics"]["train"],
        label="Training Metric",
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_ablation"]["RQS"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQS_training_validation_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training and Validation Metrics plot: {e}")
    plt.close()
