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

# Aggregate and plot training losses
try:
    losses_train = []
    for exp in experiment_data.values():
        losses_train.append(exp["synthetic_dataset"]["losses"]["train"])
    losses_train = np.array(losses_train)

    mean_losses = np.mean(losses_train, axis=0)
    std_errors = np.std(losses_train, axis=0) / np.sqrt(losses_train.shape[0])
    epochs = np.arange(1, len(mean_losses) + 1)

    plt.figure()
    plt.plot(epochs, mean_losses, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_losses - std_errors,
        mean_losses + std_errors,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")

# Aggregate and plot training CODS metrics
try:
    metrics_train = []
    for exp in experiment_data.values():
        metrics_train.append(exp["synthetic_dataset"]["metrics"]["train"])
    metrics_train = np.array(metrics_train)

    mean_metrics = np.mean(metrics_train, axis=0)
    std_errors = np.std(metrics_train, axis=0) / np.sqrt(metrics_train.shape[0])
    epochs = np.arange(1, len(mean_metrics) + 1)

    plt.figure()
    plt.plot(epochs, mean_metrics, label="Mean Training CODS", color="orange")
    plt.fill_between(
        epochs,
        mean_metrics - std_errors,
        mean_metrics + std_errors,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Training CODS over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_training_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training CODS plot: {e}")
