import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plotting training and validation losses
try:
    losses = experiment_data["weight_decay_tuning"]["synthetic_data"]["losses"]
    plt.figure()
    plt.plot(losses["train"], label="Training Loss")
    plt.plot(losses["val"], label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plotting training and validation metrics
try:
    metrics = experiment_data["weight_decay_tuning"]["synthetic_data"]["metrics"]
    plt.figure()
    plt.plot(metrics["train"], label="Training Metric")
    plt.plot(metrics["val"], label="Validation Metric")
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_training_validation_metrics.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
