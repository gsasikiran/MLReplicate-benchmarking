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

# Plot training losses
try:
    losses_train = experiment_data["dropout_tuning"]["synthetic_dataset"]["losses"][
        "train"
    ]
    epochs = np.arange(1, len(losses_train) + 1)
    plt.figure()
    plt.plot(epochs, losses_train, label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")

# Plot training CODS metrics
try:
    metrics_train = experiment_data["dropout_tuning"]["synthetic_dataset"]["metrics"][
        "train"
    ]
    epochs = np.arange(1, len(metrics_train) + 1)
    plt.figure()
    plt.plot(epochs, metrics_train, label="Training CODS", color="orange")
    plt.title("Training CODS over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training CODS plot: {e}")
