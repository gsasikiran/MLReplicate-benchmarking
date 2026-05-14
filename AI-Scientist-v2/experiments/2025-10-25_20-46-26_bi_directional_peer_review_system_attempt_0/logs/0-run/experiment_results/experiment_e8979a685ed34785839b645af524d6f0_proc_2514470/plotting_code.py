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
    epochs = list(
        range(
            1,
            len(
                experiment_data["hyperparam_tuning_num_epochs"]["feedback_dataset"][
                    "losses"
                ]["train"]
            )
            + 1,
        )
    )
    train_losses = experiment_data["hyperparam_tuning_num_epochs"]["feedback_dataset"][
        "losses"
    ]["train"]
    val_losses = experiment_data["hyperparam_tuning_num_epochs"]["feedback_dataset"][
        "losses"
    ]["val"]

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "Feedback_Dataset_Training_Validation_Losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot for losses: {e}")
    plt.close()

try:
    train_metrics = experiment_data["hyperparam_tuning_num_epochs"]["feedback_dataset"][
        "metrics"
    ]["train"]

    plt.figure()
    plt.plot(epochs, train_metrics, label="Training Metrics")
    plt.title("Training Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Feedback_Dataset_Training_Metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for metrics: {e}")
    plt.close()
