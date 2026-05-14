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

for norm_name, data in experiment_data["Normalization_Techniques_Comparison"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{norm_name} Normalization - Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{norm_name}_losses_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {norm_name} losses: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["val"], label="PWIS Metric")
        plt.title(f"{norm_name} Normalization - PWIS Metric")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{norm_name}_PWIS_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {norm_name} PWIS: {e}")
        plt.close()
