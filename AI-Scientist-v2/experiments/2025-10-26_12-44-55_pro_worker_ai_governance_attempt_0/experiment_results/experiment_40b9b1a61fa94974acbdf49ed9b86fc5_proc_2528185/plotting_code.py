import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plotting training and validation losses
try:
    losses = experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
        "losses"
    ]
    epochs = np.arange(1, len(losses["train"]) + 1)

    plt.figure()
    plt.plot(epochs, losses["train"], label="Training Loss")
    plt.plot(epochs, losses["val"], label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ablation_job_displacement_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation losses plot: {e}")
    plt.close()

# Plotting PWIS
try:
    metrics = experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
        "metrics"
    ]

    plt.figure()
    plt.plot(epochs, metrics["val"], label="PWIS")
    plt.title("Validation Pro-Worker Impact Score (PWIS)")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ablation_job_displacement_pwis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()

# Plotting predictions against ground truth
try:
    predictions = experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
        "predictions"
    ]
    ground_truth = experiment_data["ablation_job_displacement"][
        "synthetic_worker_data"
    ]["ground_truth"]

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        color="red",
    )  # Perfect prediction line
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "ablation_job_displacement_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
