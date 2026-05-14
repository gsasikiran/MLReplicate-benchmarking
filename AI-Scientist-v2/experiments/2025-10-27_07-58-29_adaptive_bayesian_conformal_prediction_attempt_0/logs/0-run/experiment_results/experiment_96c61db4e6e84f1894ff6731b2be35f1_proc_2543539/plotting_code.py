import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    losses = experiment_data["hyperparam_tuning_batch_size"]["synthetic_data"]["losses"]
    plt.figure()
    plt.plot(losses["train"], label="Train Loss")
    plt.plot(losses["val"], label="Validation Loss")
    plt.title("Loss Curves: Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot predictions vs ground truth
try:
    predictions = experiment_data["hyperparam_tuning_batch_size"]["synthetic_data"][
        "predictions"
    ]
    ground_truth = experiment_data["hyperparam_tuning_batch_size"]["synthetic_data"][
        "ground_truth"
    ]

    plt.figure()
    for i, pred in enumerate(predictions[::20]):  # Plot every 20th prediction
        plt.scatter(
            range(len(pred)), pred, label=f"Predictions (Epoch {i * 20})", alpha=0.5
        )

    plt.scatter(
        range(len(ground_truth[0])),
        ground_truth[0],
        label="Ground Truth",
        color="red",
        alpha=0.5,
    )
    plt.title("Predictions vs Ground Truth: Synthetic Data")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "predictions_vs_ground_truth_synthetic_data.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
