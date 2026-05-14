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
    plt.plot(experiment_data["peer_review"]["losses"]["train"], label="Training Loss")
    plt.title("Peer Review Experiment - Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["peer_review"]["losses"]["val"],
        label="Validation Loss",
        color="orange",
    )
    plt.title("Peer Review Experiment - Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["peer_review"]["metrics"]["train"],
        label="RQI (Train)",
        color="green",
    )
    plt.title("Peer Review Experiment - RQI Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("RQI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_rqi_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQI metrics plot: {e}")
    plt.close()
