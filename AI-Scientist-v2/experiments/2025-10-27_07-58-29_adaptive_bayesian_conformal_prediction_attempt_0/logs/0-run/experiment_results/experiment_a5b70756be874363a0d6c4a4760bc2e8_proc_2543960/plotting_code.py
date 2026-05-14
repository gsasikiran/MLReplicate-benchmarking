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
    # Plot training and validation loss
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_momentum"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_momentum"]["synthetic_data"]["losses"][
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
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    # Plot predictions vs ground truth
    predictions = np.array(
        experiment_data["hyperparam_tuning_momentum"]["synthetic_data"]["predictions"]
    ).mean(axis=0)
    ground_truth = np.array(
        experiment_data["hyperparam_tuning_momentum"]["synthetic_data"]["ground_truth"]
    ).mean(axis=0)

    plt.figure()
    plt.scatter(ground_truth, predictions)
    plt.plot(
        [ground_truth.min(), ground_truth.max()],
        [ground_truth.min(), ground_truth.max()],
        "r--",
    )
    plt.title("Predictions vs Ground Truth for Synthetic Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Model Predictions")
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
