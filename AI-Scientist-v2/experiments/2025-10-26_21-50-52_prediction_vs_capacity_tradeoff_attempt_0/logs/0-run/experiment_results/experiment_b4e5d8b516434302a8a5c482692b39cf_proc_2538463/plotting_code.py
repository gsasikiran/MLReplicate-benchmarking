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
        experiment_data["hyperparam_tuning_momentum"]["synthetic_dataset"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.title("Training Loss Curve - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_curve_synthetic.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_momentum"]["synthetic_dataset"]["metrics"][
            "train"
        ],
        label="Train Accuracy",
    )
    plt.title("Training Accuracy Curve - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_accuracy_curve_synthetic.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
