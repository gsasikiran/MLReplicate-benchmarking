import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for batch_size, data in experiment_data["batch_size_tuning"].items():
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    metrics_train = data["metrics"]["train"]
    epochs = range(1, len(train_losses) + 1)

    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{batch_size} Training and Validation Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{batch_size}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {batch_size}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(epochs, metrics_train, label="Training RAS Metric")
        plt.title(f"{batch_size} Training RAS Metric over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Synthetic RAS Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{batch_size}_RAS_metric_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RAS metric plot for {batch_size}: {e}")
        plt.close()
