import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation loss
try:
    train_losses = experiment_data["early_stopping"]["synthetic_dataset"]["losses"][
        "train"
    ]
    val_losses = experiment_data["early_stopping"]["synthetic_dataset"]["losses"]["val"]
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss Over Epochs (Synthetic Dataset)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_over_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot training and validation accuracy just like above
try:
    train_accuracy = experiment_data["early_stopping"]["synthetic_dataset"]["metrics"][
        "train"
    ]
    val_accuracy = experiment_data["early_stopping"]["synthetic_dataset"]["metrics"][
        "val"
    ]

    plt.figure()
    plt.plot(epochs, train_accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy Over Epochs (Synthetic Dataset)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_accuracy_over_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
