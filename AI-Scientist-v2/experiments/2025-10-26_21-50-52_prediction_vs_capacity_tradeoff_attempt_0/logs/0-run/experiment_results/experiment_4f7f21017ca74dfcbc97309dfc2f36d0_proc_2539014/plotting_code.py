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

try:
    train_losses = experiment_data["early_stopping"]["synthetic_dataset"]["losses"][
        "train"
    ]
    val_losses = experiment_data["early_stopping"]["synthetic_dataset"]["losses"]["val"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    train_accuracies = experiment_data["early_stopping"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    val_accuracies = experiment_data["early_stopping"]["synthetic_dataset"]["metrics"][
        "val"
    ]
    plt.figure()
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
