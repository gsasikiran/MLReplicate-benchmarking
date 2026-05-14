import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    epochs = np.arange(
        1,
        len(
            experiment_data["loss_function_ablation"]["synthetic_worker_data"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["loss_function_ablation"]["synthetic_worker_data"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["loss_function_ablation"]["synthetic_worker_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["loss_function_ablation"]["synthetic_worker_data"]["metrics"][
            "val"
        ],
        label="WIS",
    )
    plt.title("Validation Metric (WIS) Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Worker Impact Score (WIS)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_wis_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating WIS plot: {e}")
    plt.close()

try:
    predictions = np.array(
        experiment_data["loss_function_ablation"]["synthetic_worker_data"][
            "predictions"
        ]
    )
    ground_truth = np.array(
        experiment_data["loss_function_ablation"]["synthetic_worker_data"][
            "ground_truth"
        ]
    )
    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Validation Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.axline((0, 0), slope=1, color="r", linestyle="--")  # 45-degree reference line
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
