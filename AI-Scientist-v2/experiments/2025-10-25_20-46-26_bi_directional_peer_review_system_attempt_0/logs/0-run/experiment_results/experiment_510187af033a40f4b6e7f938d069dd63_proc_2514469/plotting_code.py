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
    metrics_train = experiment_data["hyperparam_tuning_activation_function"][
        "feedback_dataset"
    ]["metrics"]["train"]
    losses_train = experiment_data["hyperparam_tuning_activation_function"][
        "feedback_dataset"
    ]["losses"]["train"]
    losses_val = experiment_data["hyperparam_tuning_activation_function"][
        "feedback_dataset"
    ]["losses"]["val"]

    plt.figure()
    plt.plot(range(1, len(losses_train) + 1), losses_train, label="Training Loss")
    plt.plot(range(1, len(losses_val) + 1), losses_val, label="Validation Loss")
    plt.title("Training and Validation Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "feedback_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(range(1, len(metrics_train) + 1), metrics_train, label="Training Metric")
    plt.title("Training Metrics over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "feedback_dataset_training_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
