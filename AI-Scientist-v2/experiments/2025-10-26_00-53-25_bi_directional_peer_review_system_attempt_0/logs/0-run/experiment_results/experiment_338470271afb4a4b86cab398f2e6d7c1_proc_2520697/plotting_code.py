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

for distribution in ["normal", "uniform", "exponential"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["data_dist_variation"][distribution]["metrics"]["train"],
            label="Train Metric",
        )
        plt.plot(
            experiment_data["data_dist_variation"][distribution]["metrics"]["val"],
            label="Validation Metric",
        )
        plt.title(f"{distribution.capitalize()} Distribution - Metrics Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{distribution}_metrics_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {distribution}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["data_dist_variation"][distribution]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["data_dist_variation"][distribution]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{distribution.capitalize()} Distribution - Losses Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{distribution}_losses_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating losses plot for {distribution}: {e}")
        plt.close()
