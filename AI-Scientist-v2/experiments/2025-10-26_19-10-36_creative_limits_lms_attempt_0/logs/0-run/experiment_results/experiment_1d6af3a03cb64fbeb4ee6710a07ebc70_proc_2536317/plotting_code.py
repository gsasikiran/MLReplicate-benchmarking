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
    losses = experiment_data["learning_rate_tuning"]["synthetic_dataset"]["losses"][
        "train"
    ]
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss over Epochs - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_synthetic_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot CODS metrics
try:
    cods = experiment_data["learning_rate_tuning"]["synthetic_dataset"]["metrics"][
        "train"
    ]
    plt.figure()
    plt.plot(cods, label="CODS Metric")
    plt.title("CODS Metric over Epochs - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "cods_metric_synthetic_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CODS metric plot: {e}")
    plt.close()
