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
    dropout_tuning_data = experiment_data["dropout_tuning"]["synthetic_data"]
    epochs = range(1, len(dropout_tuning_data["losses"]["train"]) + 1)

    plt.figure()
    plt.plot(epochs, dropout_tuning_data["losses"]["train"], label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Loss_Synthetic_Data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    rqs = dropout_tuning_data["metrics"]["train"]

    plt.figure()
    plt.plot(epochs, rqs, label="RQS", color="orange")
    plt.title("RQS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQS_Synthetic_Data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQS plot: {e}")
    plt.close()
