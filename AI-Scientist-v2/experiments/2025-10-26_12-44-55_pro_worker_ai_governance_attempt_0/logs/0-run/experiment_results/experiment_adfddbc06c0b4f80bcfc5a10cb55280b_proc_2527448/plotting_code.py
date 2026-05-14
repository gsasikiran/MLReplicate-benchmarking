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
    plt.title("Loss Curves for Synthetic Worker Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    predictions = experiment_data["hyperparam_tuning_type_1"]["synthetic_worker_data"][
        "predictions"
    ]
    ground_truth = experiment_data["hyperparam_tuning_type_1"]["synthetic_worker_data"][
        "ground_truth"
    ]
    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Predictions vs Ground Truth for Synthetic Worker Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.grid()
    plt.savefig(
        os.path.join(working_dir, "synthetic_worker_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
