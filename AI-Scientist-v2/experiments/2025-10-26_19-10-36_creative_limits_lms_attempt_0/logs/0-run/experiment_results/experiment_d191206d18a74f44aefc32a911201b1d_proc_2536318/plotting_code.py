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
    epochs = range(
        1,
        len(
            experiment_data["weight_decay_tuning"]["synthetic_dataset"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    train_losses = experiment_data["weight_decay_tuning"]["synthetic_dataset"][
        "losses"
    ]["train"]
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plotting CODS metrics
try:
    cods_metrics = experiment_data["weight_decay_tuning"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    plt.figure()
    plt.plot(epochs, cods_metrics, label="CODS Metric")
    plt.title("CODS Metric over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_cods_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CODS metric plot: {e}")
    plt.close()
