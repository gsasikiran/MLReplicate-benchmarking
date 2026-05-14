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
        experiment_data["baseline"]["synthetic_data"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["baseline"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves on Synthetic Data - Baseline")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_baseline.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot for baseline: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss Curves on Synthetic Data - With Scheduler")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_with_scheduler.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot for scheduler: {e}")
    plt.close()

try:
    for idx, predictions in enumerate(
        experiment_data["baseline"]["synthetic_data"]["predictions"]
    ):
        if idx % 20 == 0:  # Plotting at intervals
            plt.figure()
            plt.scatter(
                experiment_data["baseline"]["synthetic_data"]["ground_truth"][0],
                predictions,
                label="Predictions",
            )
            plt.title(f"Predictions vs Ground Truth - Baseline (Epoch {idx})")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"predictions_baseline_epoch_{idx}.png")
            )
            plt.close()
except Exception as e:
    print(f"Error creating predictions plot for baseline: {e}")
    plt.close()

try:
    for idx, predictions in enumerate(
        experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"][
            "predictions"
        ]
    ):
        if idx % 20 == 0:  # Plotting at intervals
            plt.figure()
            plt.scatter(
                experiment_data["ablation_learning_rate_scheduler"]["synthetic_data"][
                    "ground_truth"
                ][0],
                predictions,
                label="Predictions",
            )
            plt.title(f"Predictions vs Ground Truth - With Scheduler (Epoch {idx})")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"predictions_with_scheduler_epoch_{idx}.png")
            )
            plt.close()
except Exception as e:
    print(f"Error creating predictions plot for scheduler: {e}")
    plt.close()
