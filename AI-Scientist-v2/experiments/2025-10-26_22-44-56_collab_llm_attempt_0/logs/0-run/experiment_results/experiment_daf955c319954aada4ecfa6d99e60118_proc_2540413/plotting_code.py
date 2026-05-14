import matplotlib.pyplot as plt
import numpy as np
import os

# Preparation
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training losses
for complexity in experiment_data["multi_dataset_robustness"]:
    try:
        losses = experiment_data["multi_dataset_robustness"][complexity]["losses"][
            "train"
        ]
        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.title(f"Training Loss for {complexity} Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_{complexity}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {complexity} losses: {e}")
        plt.close()

# Plot training metrics
for complexity in experiment_data["multi_dataset_robustness"]:
    try:
        metrics = experiment_data["multi_dataset_robustness"][complexity]["metrics"][
            "train"
        ]
        plt.figure()
        plt.plot(metrics, label="Training CIS")
        plt.title(f"Training CIS for {complexity} Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("CIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_metrics_{complexity}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {complexity} metrics: {e}")
        plt.close()
