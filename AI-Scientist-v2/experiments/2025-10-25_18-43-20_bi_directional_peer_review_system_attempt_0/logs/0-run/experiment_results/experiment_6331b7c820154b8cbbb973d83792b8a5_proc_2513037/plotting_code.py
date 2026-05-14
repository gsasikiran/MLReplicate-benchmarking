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
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_optimizer"]["RQI_experiment"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_optimizer"]["RQI_experiment"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss Curves for RQI Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQI_Experiment_Loss_Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for loss curves: {e}")
    plt.close()
