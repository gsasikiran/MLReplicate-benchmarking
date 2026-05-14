import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    plt.figure()
    plt.plot(experiment_data["pro_worker"]["losses"]["train"], label="Training Loss")
    plt.plot(experiment_data["pro_worker"]["losses"]["val"], label="Validation Loss")
    plt.title("Training and Validation Losses for Pro-Worker Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "pro_worker_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for losses: {e}")
    plt.close()
