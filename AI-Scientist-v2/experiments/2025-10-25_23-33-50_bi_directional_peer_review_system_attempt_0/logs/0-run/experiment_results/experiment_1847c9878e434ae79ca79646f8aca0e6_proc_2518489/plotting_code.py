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

try:
    # Plotting training and validation losses
    plt.figure()
    plt.plot(experiment_data["peer_review"]["losses"]["train"], label="Training Loss")
    plt.plot(experiment_data["peer_review"]["losses"]["val"], label="Validation Loss")
    plt.title("Loss Curves for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

try:
    # Plotting Reviewer Impact Score (RIS)
    plt.figure()
    plt.plot(experiment_data["peer_review"]["RIS"], label="Reviewer Impact Score")
    plt.title("Reviewer Impact Score Over Epochs for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Reviewer Impact Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_ris_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RIS plot: {e}")
    plt.close()

try:
    # Plotting RQI for validation
    plt.figure()
    plt.plot(experiment_data["peer_review"]["metrics"]["train"], label="RQI (Train)")
    plt.title("Review Quality Indicator for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("RQI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_rqi_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQI plot: {e}")
    plt.close()
