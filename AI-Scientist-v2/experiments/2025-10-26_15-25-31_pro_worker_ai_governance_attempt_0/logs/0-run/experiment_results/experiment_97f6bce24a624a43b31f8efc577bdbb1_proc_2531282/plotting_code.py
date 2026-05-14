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
    # Plot training and validation losses
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot training and validation metrics
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["metrics"][
            "train"
        ],
        label="Training Accuracy",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_batch_norm"]["synthetic_data"]["metrics"][
            "val"
        ],
        label="Validation Accuracy",
    )
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
