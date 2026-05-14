import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Generate plots
for act_name, data in experiment_data["feature_impact_ablation"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves - Activation Function: {act_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"Loss_Curves_{act_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {act_name}: {e}")
        plt.close()
