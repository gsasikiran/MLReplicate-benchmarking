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

scalers = experiment_data["input_feature_scaling_variation"]

# Plot Training and Validation Loss
for scaler_name in scalers.keys():
    try:
        plt.figure()
        plt.plot(scalers[scaler_name]["losses"]["train"], label="Training Loss")
        plt.plot(scalers[scaler_name]["losses"]["val"], label="Validation Loss")
        plt.title(f"{scaler_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{scaler_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {scaler_name} loss: {e}")
        plt.close()

# Plot Validation Metrics (PWIS)
for scaler_name in scalers.keys():
    try:
        plt.figure()
        plt.plot(scalers[scaler_name]["metrics"]["val"], label="PWIS")
        plt.title(f"{scaler_name} Validation Metrics (PWIS)")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{scaler_name}_validation_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {scaler_name} validation metrics: {e}")
        plt.close()
