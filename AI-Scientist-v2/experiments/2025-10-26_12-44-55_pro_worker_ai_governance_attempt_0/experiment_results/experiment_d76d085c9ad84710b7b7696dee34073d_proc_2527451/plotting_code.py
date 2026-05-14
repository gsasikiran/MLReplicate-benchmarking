import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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
        experiment_data["hyperparam_tuning_type_1"]["synthetic_worker_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_worker_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_worker_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

try:
    # Plot predictions vs ground truth
    predictions = experiment_data["hyperparam_tuning_type_1"]["synthetic_worker_data"][
        "predictions"
    ]
    ground_truth = experiment_data["hyperparam_tuning_type_1"]["synthetic_worker_data"][
        "ground_truth"
    ]

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        color="red",
    )  # line of equality
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "synthetic_worker_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
