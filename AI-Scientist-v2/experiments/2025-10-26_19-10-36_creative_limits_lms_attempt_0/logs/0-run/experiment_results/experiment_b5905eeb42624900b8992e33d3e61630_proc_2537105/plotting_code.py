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

# Plot Training Loss
try:
    metrics = experiment_data["activation_variation"]["synthetic_dataset"]["losses"][
        "train"
    ]
    epochs = range(1, len(metrics) + 1)
    plt.figure()
    plt.plot(epochs, metrics, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot CODS
try:
    cods_metrics = experiment_data["activation_variation"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    epochs = range(1, len(cods_metrics) + 1)
    plt.figure()
    plt.plot(epochs, cods_metrics, label="CODS Metrics", color="orange")
    plt.title("CODS Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_cods_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CODS metrics plot: {e}")
    plt.close()
