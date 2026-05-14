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
        experiment_data["training_epochs_ablation"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["training_epochs_ablation"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Synthetic Data - Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["training_epochs_ablation"]["synthetic_data"]["metrics"][
            "train"
        ],
        label="Training Accuracy",
    )
    plt.plot(
        experiment_data["training_epochs_ablation"]["synthetic_data"]["metrics"]["val"],
        label="Validation Accuracy",
    )
    plt.title("Synthetic Data - Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()
