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

for correlation_level in experiment_data["input_feature_correlation_analysis"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["input_feature_correlation_analysis"][correlation_level][
                "losses"
            ]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["input_feature_correlation_analysis"][correlation_level][
                "losses"
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for {correlation_level.capitalize()} Correlation Level")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{correlation_level}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {correlation_level}: {e}")
        plt.close()
