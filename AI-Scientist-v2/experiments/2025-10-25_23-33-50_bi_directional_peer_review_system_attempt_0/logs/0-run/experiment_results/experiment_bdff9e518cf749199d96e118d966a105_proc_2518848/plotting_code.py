import matplotlib.pyplot as plt
import numpy as np
import os

# Setting up working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training and validation losses
try:
    plt.figure()
    plt.plot(
        experiment_data["batch_size_variability"]["peer_review"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["batch_size_variability"]["peer_review"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_training_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

# Plotting RQS
try:
    plt.figure()
    plt.plot(
        experiment_data["batch_size_variability"]["peer_review"]["metrics"]["RQS"],
        label="RQS",
    )
    plt.title("RQS Over Epochs for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("RQS Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQS plot: {e}")
    plt.close()
