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

# Training/Validation Losses Plot
try:
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

# Comparison of Losses Between Datasets
try:
    plt.figure()
    plt.plot(experiment_data["dataset_1"]["losses"], label="Dataset 1 Loss")
    plt.plot(experiment_data["dataset_2"]["losses"], label="Dataset 2 Loss")
    plt.plot(experiment_data["dataset_3"]["losses"], label="Dataset 3 Loss")
    plt.title("Loss Comparison Among Different Datasets")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dataset_loss_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dataset loss comparison plot: {e}")
    plt.close()
