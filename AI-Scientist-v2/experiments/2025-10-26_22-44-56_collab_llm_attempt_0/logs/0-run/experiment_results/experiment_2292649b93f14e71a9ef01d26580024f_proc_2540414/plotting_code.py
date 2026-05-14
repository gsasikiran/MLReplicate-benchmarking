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

for structure, data in experiment_data["prompt_structure_variation"].items():
    # Training Loss Plot
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.title(f"Training Loss for {structure}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"Training_Loss_{structure}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Training Loss plot for {structure}: {e}")
        plt.close()

    # UES Metric Plot
    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="UES Metric", color="orange")
        plt.title(f"UES Metric for {structure}")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"UES_Metric_{structure}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating UES Metric plot for {structure}: {e}")
        plt.close()
