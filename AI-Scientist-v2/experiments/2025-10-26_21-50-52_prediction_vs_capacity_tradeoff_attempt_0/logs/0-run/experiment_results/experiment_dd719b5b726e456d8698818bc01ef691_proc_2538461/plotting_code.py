import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Training Loss Plot
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_lr"]["synthetic_dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Training Accuracy Plot
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_lr"]["synthetic_dataset"]["metrics"][
            "train"
        ],
        label="Training Accuracy",
    )
    plt.title("Training Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_accuracy_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
