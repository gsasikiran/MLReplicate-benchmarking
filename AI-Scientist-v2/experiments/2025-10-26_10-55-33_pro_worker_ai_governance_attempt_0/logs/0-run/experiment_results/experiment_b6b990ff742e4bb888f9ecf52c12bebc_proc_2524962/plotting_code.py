import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for model_name, data in experiment_data["depth_variation"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Train Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{model_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {model_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["val"], label="Validation PWIS")
        plt.title(f"{model_name} Validation PWIS")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_name}_PWIS.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PWIS plot for {model_name}: {e}")
        plt.close()
