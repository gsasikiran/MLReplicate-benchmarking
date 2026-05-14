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
        experiment_data["noise_injection"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["noise_injection"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Losses for Noise Injection Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "noise_injection_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(experiment_data["noise_injection"]["metrics"]["train"], label="RQS")
    plt.title("Reviewer Quality Score (RQS) for Noise Injection Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "noise_injection_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQS plot: {e}")
    plt.close()
