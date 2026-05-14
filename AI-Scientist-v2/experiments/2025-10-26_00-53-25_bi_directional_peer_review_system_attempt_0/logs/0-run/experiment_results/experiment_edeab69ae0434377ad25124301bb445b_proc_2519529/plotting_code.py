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
    plt.plot(
        experiment_data["synthetic_reviews"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["synthetic_reviews"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "synthetic_reviews_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for losses: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_reviews"]["metrics"]["val"], label="Validation RQS"
    )
    plt.title("Validation RQS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Validation RQS")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "synthetic_reviews_validation_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for validation RQS: {e}")
    plt.close()
