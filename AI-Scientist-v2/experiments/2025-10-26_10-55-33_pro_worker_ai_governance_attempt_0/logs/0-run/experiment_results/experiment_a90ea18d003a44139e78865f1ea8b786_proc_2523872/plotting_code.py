import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extracting values
try:
    train_losses = experiment_data["hyperparam_tuning_weight_decay"][
        "synthetic_dataset"
    ]["losses"]["train"]
    val_losses = experiment_data["hyperparam_tuning_weight_decay"]["synthetic_dataset"][
        "losses"
    ]["val"]
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Losses - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    val_metrics = experiment_data["hyperparam_tuning_weight_decay"][
        "synthetic_dataset"
    ]["metrics"]["val"]

    plt.figure()
    plt.plot(epochs, val_metrics, label="PWIS (Higher is Better)")
    plt.title("Validation PWIS - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_validation_pwis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
