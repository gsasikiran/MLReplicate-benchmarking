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

# Plot aggregated training losses
try:
    losses_train = experiment_data["dropout_tuning"]["synthetic_dataset"]["losses"][
        "train"
    ]
    epochs = np.arange(1, len(losses_train) + 1)
    mean_losses = np.mean(losses_train)
    se_losses = np.std(losses_train) / np.sqrt(len(losses_train))
    plt.figure()
    plt.errorbar(
        epochs,
        losses_train,
        yerr=se_losses,
        label="Training Loss (mean ± SE)",
        fmt="-o",
    )
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_loss_with_error.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")

# Plot aggregated training CODS metrics
try:
    metrics_train = experiment_data["dropout_tuning"]["synthetic_dataset"]["metrics"][
        "train"
    ]
    epochs = np.arange(1, len(metrics_train) + 1)
    mean_metrics = np.mean(metrics_train)
    se_metrics = np.std(metrics_train) / np.sqrt(len(metrics_train))
    plt.figure()
    plt.errorbar(
        epochs,
        metrics_train,
        yerr=se_metrics,
        label="Training CODS (mean ± SE)",
        fmt="-o",
        color="orange",
    )
    plt.title("Training CODS over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_cods_with_error.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training CODS plot: {e}")
