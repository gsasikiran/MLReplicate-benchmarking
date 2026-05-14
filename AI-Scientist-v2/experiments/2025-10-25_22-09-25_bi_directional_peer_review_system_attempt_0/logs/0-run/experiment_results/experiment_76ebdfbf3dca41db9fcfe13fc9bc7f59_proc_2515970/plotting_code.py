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

# Plot training loss
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["train"], label="Training Loss"
    )
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training metrics (RQS)
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["metrics"]["train"],
        label="Training RQS",
        color="orange",
    )
    plt.title("Training RQS Curve")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training RQS plot: {e}")
    plt.close()
