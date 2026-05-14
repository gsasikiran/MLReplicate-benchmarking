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

for act_name, data in experiment_data["activation_function_variation"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Activation Function: {act_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{act_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {act_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["val"], label="WWBI Metric")
        plt.title(f"WWBI Metric for Activation Function: {act_name}")
        plt.xlabel("Epochs")
        plt.ylabel("WWBI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"wwbi_metric_{act_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating WWBI plot for {act_name}: {e}")
        plt.close()
