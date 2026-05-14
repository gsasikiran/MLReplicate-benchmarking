import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    # Plot Training and Validation Loss
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_dropout"]["synthetic_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_dropout"]["synthetic_dataset"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

try:
    # Plot Validation PWIS Metric
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_dropout"]["synthetic_dataset"]["metrics"][
            "val"
        ],
        label="PWIS (Validation)",
    )
    plt.title("Performance Metric (PWIS) over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_val_pwis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
