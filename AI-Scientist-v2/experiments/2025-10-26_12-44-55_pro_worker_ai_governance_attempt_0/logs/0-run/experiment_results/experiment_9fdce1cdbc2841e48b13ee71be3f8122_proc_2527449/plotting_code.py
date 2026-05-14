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

# Plot training and validation losses
try:
    epochs = range(
        1,
        len(
            experiment_data["batch_size_tuning"]["synthetic_worker_data"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_worker_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_worker_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses for Synthetic Worker Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot validation metrics (WIS)
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_worker_data"]["metrics"]["val"],
        label="WIS",
    )
    plt.title("Validation Metrics (WIS) for Synthetic Worker Data")
    plt.xlabel("Epochs")
    plt.ylabel("WIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_WIS_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating WIS plot: {e}")
    plt.close()
