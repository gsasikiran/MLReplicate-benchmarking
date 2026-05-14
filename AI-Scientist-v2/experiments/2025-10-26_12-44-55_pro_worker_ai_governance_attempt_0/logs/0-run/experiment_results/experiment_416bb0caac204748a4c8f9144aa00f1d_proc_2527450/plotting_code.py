import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()

    epochs = [10, 20, 30, 40, 50]
    train_losses = experiment_data["hyperparam_tuning_num_epochs"][
        "synthetic_worker_data"
    ]["losses"]["train"]
    val_losses = experiment_data["hyperparam_tuning_num_epochs"][
        "synthetic_worker_data"
    ]["losses"]["val"]

    # Training Loss Plot
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss Curves for Synthetic Worker Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_worker_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating Training Loss plot: {e}")
    plt.close()
