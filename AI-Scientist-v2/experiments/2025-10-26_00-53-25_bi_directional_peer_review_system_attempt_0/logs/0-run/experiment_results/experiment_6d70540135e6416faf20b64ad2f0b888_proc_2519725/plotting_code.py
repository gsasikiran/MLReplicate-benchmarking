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

try:
    metrics_train = experiment_data["hyperparam_tuning_batch_size"]["RQS"]["metrics"][
        "train"
    ]
    metrics_val = experiment_data["hyperparam_tuning_batch_size"]["RQS"]["metrics"][
        "val"
    ]

    epochs = np.arange(1, len(metrics_train) + 1)
    plt.figure()
    plt.plot(epochs, metrics_train, label="Training Metric")
    plt.plot(epochs, metrics_val, label="Validation Metric")
    plt.title("Training and Validation Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQS_training_validation_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

try:
    losses_train = experiment_data["hyperparam_tuning_batch_size"]["RQS"]["losses"][
        "train"
    ]
    losses_val = experiment_data["hyperparam_tuning_batch_size"]["RQS"]["losses"]["val"]

    plt.figure()
    plt.plot(epochs, losses_train, label="Training Loss")
    plt.plot(epochs, losses_val, label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQS_training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating losses plot: {e}")
    plt.close()
