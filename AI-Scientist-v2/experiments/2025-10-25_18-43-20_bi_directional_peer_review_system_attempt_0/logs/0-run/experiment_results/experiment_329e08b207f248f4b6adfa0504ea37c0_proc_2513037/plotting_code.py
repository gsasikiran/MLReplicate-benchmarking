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
            experiment_data["hyperparam_tuning_type_1"]["RQI_experiment"]["losses"][
                "train"
            ]
        )
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_type_1"]["RQI_experiment"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_type_1"]["RQI_experiment"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for RQI Experiment")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQI_experiment_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()
