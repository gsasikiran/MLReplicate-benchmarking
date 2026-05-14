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

# Plotting Training and Validation Losses
try:
    train_losses = experiment_data["hyperparam_tuning_learning_rate"]["peer_review"][
        "losses"
    ]["train"]
    val_losses = experiment_data["hyperparam_tuning_learning_rate"]["peer_review"][
        "losses"
    ]["val"]
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plotting RQI
try:
    rqi = [
        1 - val
        for val in experiment_data["hyperparam_tuning_learning_rate"]["peer_review"][
            "losses"
        ]["val"]
    ]
    plt.figure()
    plt.plot(epochs, rqi, label="RQI")
    plt.title("Relative Quality Index (RQI)")
    plt.xlabel("Epochs")
    plt.ylabel("RQI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_rqi.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQI plot: {e}")
    plt.close()
