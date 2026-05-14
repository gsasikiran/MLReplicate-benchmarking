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

# Plot for training and validation losses
try:
    batch_sizes = experiment_data["batch_size_ablation"]["synthetic_data"][
        "batch_sizes"
    ]
    train_losses = experiment_data["batch_size_ablation"]["synthetic_data"]["losses"][
        "train"
    ]
    val_losses = experiment_data["batch_size_ablation"]["synthetic_data"]["losses"][
        "val"
    ]

    plt.figure()
    for size, train_loss, val_loss in zip(batch_sizes, train_losses, val_losses):
        plt.plot(range(len(train_loss)), train_loss, label=f"Train Loss (Batch {size})")
        plt.plot(
            range(len(val_loss)),
            val_loss,
            label=f"Val Loss (Batch {size})",
            linestyle="dashed",
        )
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "train_val_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot for training and validation accuracy
try:
    train_metrics = experiment_data["batch_size_ablation"]["synthetic_data"]["metrics"][
        "train"
    ]
    val_metrics = experiment_data["batch_size_ablation"]["synthetic_data"]["metrics"][
        "val"
    ]

    plt.figure()
    for size, train_metric, val_metric in zip(batch_sizes, train_metrics, val_metrics):
        plt.plot(
            range(len(train_metric)),
            train_metric,
            label=f"Train Accuracy (Batch {size})",
        )
        plt.plot(
            range(len(val_metric)),
            val_metric,
            label=f"Val Accuracy (Batch {size})",
            linestyle="dashed",
        )
    plt.title("Training and Validation Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "train_val_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()
