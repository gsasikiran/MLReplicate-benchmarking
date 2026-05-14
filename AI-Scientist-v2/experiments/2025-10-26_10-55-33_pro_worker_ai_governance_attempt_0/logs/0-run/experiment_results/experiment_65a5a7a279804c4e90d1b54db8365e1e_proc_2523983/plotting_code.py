import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()

    # Plot Training and Validation Losses
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_dataset"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for Loss Curves: {e}")
    plt.close()

try:
    # Plot Validation PWIS
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_dataset"]["metrics"][
            "val"
        ],
        label="Validation PWIS",
        color="orange",
    )
    plt.title("Validation PWIS Over Epochs for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_pwisk_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for PWIS: {e}")
    plt.close()
