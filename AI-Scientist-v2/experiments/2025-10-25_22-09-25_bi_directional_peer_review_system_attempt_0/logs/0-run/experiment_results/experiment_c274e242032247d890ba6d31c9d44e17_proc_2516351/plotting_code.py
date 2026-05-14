import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()
