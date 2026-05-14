import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Visually represent training metrics
try:
    dropout_rates = experiment_data["dropout_tuning"]["synthetic_data"]["metrics"]
    epochs = range(len(dropout_rates["train"]))
    plt.figure()
    plt.plot(epochs, dropout_rates["train"], label="Training Metric")
    plt.plot(epochs, dropout_rates["val"], label="Validation Metric")
    plt.title("Training and Validation Metrics per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# Visualize training losses
try:
    dropout_losses = experiment_data["dropout_tuning"]["synthetic_data"]["losses"]
    plt.figure()
    plt.plot(epochs, dropout_losses["train"], label="Training Loss")
    plt.plot(epochs, dropout_losses["val"], label="Validation Loss")
    plt.title("Training and Validation Losses per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating losses plot: {e}")
    plt.close()
