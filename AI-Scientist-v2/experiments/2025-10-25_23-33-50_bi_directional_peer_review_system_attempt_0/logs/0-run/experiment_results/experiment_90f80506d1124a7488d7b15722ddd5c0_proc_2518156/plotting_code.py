import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
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
            experiment_data["hyperparam_tuning_epoch_count"]["peer_review"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_epoch_count"]["peer_review"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_epoch_count"]["peer_review"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot RQI metric
try:
    rqi_metrics = experiment_data["hyperparam_tuning_epoch_count"]["peer_review"][
        "metrics"
    ]["train"]
    plt.figure()
    plt.plot(epochs, rqi_metrics, label="RQI", color="orange")
    plt.title("RQI Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("RQI Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_rqi.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQI plot: {e}")
    plt.close()
