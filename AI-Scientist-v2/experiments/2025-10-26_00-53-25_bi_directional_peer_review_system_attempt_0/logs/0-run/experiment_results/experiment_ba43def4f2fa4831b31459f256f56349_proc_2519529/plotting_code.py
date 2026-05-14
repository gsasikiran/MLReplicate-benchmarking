import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    epochs = range(1, len(experiment_data["peer_review"]["losses"]["train"]) + 1)
    plt.figure()
    plt.plot(
        epochs, experiment_data["peer_review"]["losses"]["train"], label="Train Loss"
    )
    plt.plot(
        epochs, experiment_data["peer_review"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs, experiment_data["peer_review"]["metrics"]["train"], label="Train RQS"
    )
    plt.plot(
        epochs, experiment_data["peer_review"]["metrics"]["val"], label="Validation RQS"
    )
    plt.title("Training and Validation RQS Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_validation_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQS plot: {e}")
    plt.close()
