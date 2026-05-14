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

try:
    epochs = range(
        len(
            experiment_data["learning_rate_tuning"]["RQI_experiment"]["losses"]["train"]
        )
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["learning_rate_tuning"]["RQI_experiment"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["learning_rate_tuning"]["RQI_experiment"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for RQI Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQI_Experiment_Loss_Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()
