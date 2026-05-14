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
    epochs = range(
        1,
        len(
            experiment_data["activation_function_study"]["synthetic_worker_data"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["activation_function_study"]["synthetic_worker_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["activation_function_study"]["synthetic_worker_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss over Epochs for Synthetic Worker Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "losses_synthetic_worker_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.scatter(
        experiment_data["activation_function_study"]["synthetic_worker_data"][
            "ground_truth"
        ],
        experiment_data["activation_function_study"]["synthetic_worker_data"][
            "predictions"
        ],
        alpha=0.5,
    )
    plt.plot(
        [
            min(
                experiment_data["activation_function_study"]["synthetic_worker_data"][
                    "ground_truth"
                ]
            ),
            max(
                experiment_data["activation_function_study"]["synthetic_worker_data"][
                    "ground_truth"
                ]
            ),
        ],
        [
            min(
                experiment_data["activation_function_study"]["synthetic_worker_data"][
                    "ground_truth"
                ]
            ),
            max(
                experiment_data["activation_function_study"]["synthetic_worker_data"][
                    "ground_truth"
                ]
            ),
        ],
        "r--",
    )
    plt.title("Predictions vs Ground Truth for Synthetic Worker Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(
            working_dir, "predictions_vs_ground_truth_synthetic_worker_data.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
