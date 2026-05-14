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

# Plotting training losses
for complexity in ["simple", "moderate", "complex"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_robustness"][complexity]["losses"]["train"],
            label="Training Loss",
        )
        plt.title(f"Training Loss for {complexity.capitalize()} Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_{complexity}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for training loss of {complexity}: {e}")
        plt.close()

# Plotting CIS metrics
for complexity in ["simple", "moderate", "complex"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_robustness"][complexity]["metrics"]["train"],
            label="CIS",
        )
        plt.title(
            f"Collaborative Interaction Score for {complexity.capitalize()} Dataset"
        )
        plt.xlabel("Epochs")
        plt.ylabel("CIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"cis_{complexity}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for CIS of {complexity}: {e}")
        plt.close()
