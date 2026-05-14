import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_learning_rate"]["synthetic_dataset"][
            "losses"
        ]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_learning_rate"]["synthetic_dataset"][
            "losses"
        ]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training-validation loss plot: {e}")
    plt.close()

# Plot validation metrics (PWIS)
try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_learning_rate"]["synthetic_dataset"][
            "metrics"
        ]["val"],
        label="PWIS",
    )
    plt.title("Validation Metrics (PWIS)")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_validation_metrics_PWIS.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
