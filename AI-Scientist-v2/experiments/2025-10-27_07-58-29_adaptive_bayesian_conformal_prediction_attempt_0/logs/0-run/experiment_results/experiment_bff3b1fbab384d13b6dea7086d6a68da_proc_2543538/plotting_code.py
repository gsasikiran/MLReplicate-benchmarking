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
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.scatter(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["ground_truth"][
            0
        ],
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["predictions"][0],
        alpha=0.5,
    )
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.plot([0, 3], [0, 3], "r--")  # y=x line for reference
    plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()

try:
    reliability = experiment_data["hyperparam_tuning_type_1"]["synthetic_data"][
        "metrics"
    ]["val"]
    plt.figure()
    plt.plot(reliability, label="Reliability Measure")
    plt.title("Reliability Measure over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Reliability")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "reliability_measure.png"))
    plt.close()
except Exception as e:
    print(f"Error creating reliability measure plot: {e}")
    plt.close()
