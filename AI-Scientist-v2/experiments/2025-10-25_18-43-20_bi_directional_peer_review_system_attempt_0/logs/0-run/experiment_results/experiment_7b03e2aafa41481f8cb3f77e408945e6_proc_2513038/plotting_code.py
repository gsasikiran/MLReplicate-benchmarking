import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    train_losses = experiment_data["dropout_rate_tuning"]["RQI_experiment"]["losses"][
        "train"
    ]
    val_losses = experiment_data["dropout_rate_tuning"]["RQI_experiment"]["losses"][
        "val"
    ]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training loss
try:
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQI_experiment_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot validation loss
try:
    plt.figure()
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Validation Loss across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQI_experiment_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
