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
    plt.plot(
        experiment_data["L2_regularization"]["synthetic_data"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["L2_regularization"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("L2 Regularization: Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "l2_regularization_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating L2 regularization loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["dropout"]["synthetic_data"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["dropout"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Dropout: Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dropout_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Dropout loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["L2_regularization"]["synthetic_data"]["metrics"]["train"],
        label="Train Metric",
    )
    plt.plot(
        experiment_data["L2_regularization"]["synthetic_data"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("L2 Regularization: Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "l2_regularization_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating L2 regularization metric plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["dropout"]["synthetic_data"]["metrics"]["train"],
        label="Train Metric",
    )
    plt.plot(
        experiment_data["dropout"]["synthetic_data"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("Dropout: Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dropout_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Dropout metric plot: {e}")
    plt.close()
