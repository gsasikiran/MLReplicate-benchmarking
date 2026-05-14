import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot Training Loss
try:
    plt.figure()
    plt.plot(
        experiment_data["num_epochs_tuning"]["synthetic_dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot CODS
try:
    plt.figure()
    plt.plot(
        experiment_data["num_epochs_tuning"]["synthetic_dataset"]["metrics"]["train"],
        label="CODS",
    )
    plt.title("Coefficient of Distinctiveness (CODS) over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CODS plot: {e}")
    plt.close()
