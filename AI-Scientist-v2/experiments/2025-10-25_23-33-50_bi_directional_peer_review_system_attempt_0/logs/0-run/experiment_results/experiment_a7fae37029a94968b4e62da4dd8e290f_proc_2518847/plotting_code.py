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

for distribution in ["uniform", "normal", "bimodal"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][distribution]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["ablation_study"][distribution]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{distribution.capitalize()} Distribution Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{distribution}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {distribution} losses: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][distribution]["metrics"]["train"],
            label="Training RQI",
        )
        plt.title(f"{distribution.capitalize()} Distribution Training Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("RQI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{distribution}_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {distribution} metrics: {e}")
        plt.close()
