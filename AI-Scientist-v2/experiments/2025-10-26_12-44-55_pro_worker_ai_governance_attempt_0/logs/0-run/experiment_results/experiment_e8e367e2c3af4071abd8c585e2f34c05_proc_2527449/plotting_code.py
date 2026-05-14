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
    # Plot training and validation losses
    plt.figure()
    epochs = list(
        range(
            len(
                experiment_data["momentum_tuning"]["synthetic_worker_data"]["losses"][
                    "train"
                ]
            )
        )
    )
    plt.plot(
        epochs,
        experiment_data["momentum_tuning"]["synthetic_worker_data"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["momentum_tuning"]["synthetic_worker_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Synthetic Worker Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    # Plot predictions vs ground truth
    plt.figure()
    plt.scatter(
        experiment_data["momentum_tuning"]["synthetic_worker_data"]["ground_truth"],
        experiment_data["momentum_tuning"]["synthetic_worker_data"]["predictions"],
        alpha=0.5,
    )
    plt.plot([0, 1], [0, 1], "--", color="red")
    plt.title("Predictions vs Ground Truth for Synthetic Worker Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
