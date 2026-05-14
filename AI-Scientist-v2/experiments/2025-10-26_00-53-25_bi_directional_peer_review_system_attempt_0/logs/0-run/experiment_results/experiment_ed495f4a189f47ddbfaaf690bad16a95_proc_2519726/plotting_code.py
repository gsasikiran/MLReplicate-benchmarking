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
    dropout_rates = experiment_data["hyperparam_tuning_dropout"]["dropout_rates"]
    train_metrics = experiment_data["hyperparam_tuning_dropout"]["metrics"]["train"]
    val_metrics = experiment_data["hyperparam_tuning_dropout"]["metrics"]["val"]

    plt.figure()
    plt.plot(dropout_rates, train_metrics, label="Training Metric")
    plt.plot(dropout_rates, val_metrics, label="Validation Metric")
    plt.title("Dropout Rate vs Metric Performance")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "dropout_rate_vs_metric_performance.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dropout rate vs metric performance plot: {e}")
    plt.close()

try:
    train_losses = experiment_data["hyperparam_tuning_dropout"]["losses"]["train"]
    val_losses = experiment_data["hyperparam_tuning_dropout"]["losses"]["val"]

    plt.figure()
    plt.plot(dropout_rates, train_losses, label="Training Loss")
    plt.plot(dropout_rates, val_losses, label="Validation Loss")
    plt.title("Dropout Rate vs Loss")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "dropout_rate_vs_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dropout rate vs loss plot: {e}")
    plt.close()
