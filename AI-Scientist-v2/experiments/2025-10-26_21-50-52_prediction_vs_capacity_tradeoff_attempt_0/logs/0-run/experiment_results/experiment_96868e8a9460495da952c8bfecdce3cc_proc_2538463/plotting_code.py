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
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"][
            "losses"
        ]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"][
            "losses"
        ]["val"],
        label="Validation Loss",
    )
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_vs_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"][
            "metrics"
        ]["train"],
        label="Train Accuracy",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_early_stopping"]["synthetic_dataset"][
            "metrics"
        ]["val"],
        label="Validation Accuracy",
    )
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_accuracy_vs_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
