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

for activation, data in experiment_data[
    "hyperparam_tuning_activation_function"
].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{activation} Activation Function")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {activation}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="Training Metric")
        plt.plot(data["metrics"]["val"], label="Validation Metric")
        plt.title(f"{activation} Activation Function Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation}_metric_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {activation}: {e}")
        plt.close()
