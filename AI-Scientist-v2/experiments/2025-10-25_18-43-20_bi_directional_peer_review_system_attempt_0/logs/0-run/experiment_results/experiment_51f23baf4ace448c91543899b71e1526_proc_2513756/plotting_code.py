import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation loss curves for each activation function
for act_name, data in experiment_data["noise_sensitivity_ablation"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves: Activation Function - {act_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {act_name}: {e}")
        plt.close()

# Plot predictions vs ground truth for each activation function
for act_name, data in experiment_data["noise_sensitivity_ablation"].items():
    try:
        plt.figure()
        plt.scatter(data["ground_truth"], data["predictions"], alpha=0.5)
        plt.title(f"Predictions vs Ground Truth: Activation Function - {act_name}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.plot([0, 1], [0, 1], "r--")  # Identity line
        plt.savefig(
            os.path.join(working_dir, f"{act_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {act_name}: {e}")
        plt.close()
