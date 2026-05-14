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

# Plotting training losses
try:
    losses = experiment_data["multiple_epochs_variation"]["synthetic_dataset"][
        "losses"
    ]["train"]
    epochs = list(range(1, len(losses) + 1))

    plt.figure()
    plt.plot(epochs, losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "Synthetic_Dataset_Training_Loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plotting CODS metrics
try:
    cods_metrics = experiment_data["multiple_epochs_variation"]["synthetic_dataset"][
        "metrics"
    ]["train"]

    plt.figure()
    plt.plot(epochs, cods_metrics, label="CODS", color="orange")
    plt.title("CODS Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "Synthetic_Dataset_CODS_Metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CODS plot: {e}")
    plt.close()
