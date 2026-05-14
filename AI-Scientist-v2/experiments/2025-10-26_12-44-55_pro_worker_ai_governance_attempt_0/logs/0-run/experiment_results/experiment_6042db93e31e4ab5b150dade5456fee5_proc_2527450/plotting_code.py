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

# Plot training and validation losses
try:
    epochs = range(
        1,
        len(
            experiment_data["hyperparam_tuning_optimizer"]["synthetic_worker_data"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    train_losses = experiment_data["hyperparam_tuning_optimizer"][
        "synthetic_worker_data"
    ]["losses"]["train"]
    val_losses = experiment_data["hyperparam_tuning_optimizer"][
        "synthetic_worker_data"
    ]["losses"]["val"]

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Synthetic Worker Data Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot Worker Impact Scores (WIS)
try:
    wis_scores = experiment_data["hyperparam_tuning_optimizer"][
        "synthetic_worker_data"
    ]["metrics"]["val"]

    plt.figure()
    plt.plot(epochs, wis_scores, label="Worker Impact Score (WIS)")
    plt.title("Synthetic Worker Data Worker Impact Score")
    plt.xlabel("Epochs")
    plt.ylabel("WIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_wis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating WIS plot: {e}")
    plt.close()
