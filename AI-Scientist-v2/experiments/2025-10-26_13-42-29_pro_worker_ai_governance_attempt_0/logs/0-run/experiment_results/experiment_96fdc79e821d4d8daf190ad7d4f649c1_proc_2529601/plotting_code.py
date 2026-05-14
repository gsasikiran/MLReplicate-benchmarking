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
    plt.figure()
    epochs = [
        1,
        2,
        3,
        4,
        5,
    ]  # Assuming you adjust to the correct epoch lengths in a real scenario
    train_losses = experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
        "losses"
    ]["train"]
    val_losses = experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
        "losses"
    ]["val"]
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Losses - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()
