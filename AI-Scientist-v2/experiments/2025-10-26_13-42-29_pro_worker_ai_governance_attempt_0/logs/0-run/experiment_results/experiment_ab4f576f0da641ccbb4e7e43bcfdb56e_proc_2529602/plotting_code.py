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

# Plotting training and validation loss for each dropout rate
for dropout_rate, data in experiment_data["dropout_rate_tuning"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Dropout Rate: {dropout_rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"dropout_rate_{dropout_rate}_loss_curve.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for dropout rate {dropout_rate}: {e}")
        plt.close()

# Plotting WWBI metrics for each dropout rate
for dropout_rate, data in experiment_data["dropout_rate_tuning"].items():
    try:
        plt.figure()
        plt.plot(data["metrics"]["val"], label="WWBI Metric")
        plt.title(f"Dropout Rate: {dropout_rate} WWBI Metric")
        plt.xlabel("Epochs")
        plt.ylabel("WWBI")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"dropout_rate_{dropout_rate}_wwbi_metric.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating WWBI metric plot for dropout rate {dropout_rate}: {e}")
        plt.close()
