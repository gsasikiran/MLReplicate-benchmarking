import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    plt.figure()
    epochs = list(
        range(
            1,
            len(
                experiment_data["hyperparam_tuning_epochs"]["synthetic_dataset"][
                    "losses"
                ]["train"]
            )
            + 1,
        )
    )
    train_losses = experiment_data["hyperparam_tuning_epochs"]["synthetic_dataset"][
        "losses"
    ]["train"]
    val_losses = experiment_data["hyperparam_tuning_epochs"]["synthetic_dataset"][
        "losses"
    ]["val"]
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    pwis_metrics = experiment_data["hyperparam_tuning_epochs"]["synthetic_dataset"][
        "metrics"
    ]["val"]
    plt.plot(epochs, pwis_metrics, label="PWIS Metric", color="orange")
    plt.title("Validation PWIS Metric Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_pwis_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
