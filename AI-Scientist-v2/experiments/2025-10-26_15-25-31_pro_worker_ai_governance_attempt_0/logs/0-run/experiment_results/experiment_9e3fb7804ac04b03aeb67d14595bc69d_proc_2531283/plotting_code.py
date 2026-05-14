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

# Plotting Training and Validation Loss
try:
    epochs = len(
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_data"]["losses"][
            "train"
        ]
    )
    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        range(1, epochs + 1),
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plotting Training and Validation Metrics (Accuracy)
try:
    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_data"]["metrics"][
            "train"
        ],
        label="Training Accuracy",
    )
    plt.plot(
        range(1, epochs + 1),
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_data"]["metrics"][
            "val"
        ],
        label="Validation Accuracy",
    )
    plt.title("Accuracy Curves for Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
