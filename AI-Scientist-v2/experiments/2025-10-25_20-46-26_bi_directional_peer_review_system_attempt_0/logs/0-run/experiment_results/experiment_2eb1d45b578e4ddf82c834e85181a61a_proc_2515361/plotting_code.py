import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    epochs = range(
        1,
        len(
            experiment_data["input_noise_injection"]["FeedbackDataset"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    train_losses = experiment_data["input_noise_injection"]["FeedbackDataset"][
        "losses"
    ]["train"]
    val_losses = experiment_data["input_noise_injection"]["FeedbackDataset"]["losses"][
        "val"
    ]

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss Curves for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot training metrics (synthetic RAS)
try:
    metrics = experiment_data["input_noise_injection"]["FeedbackDataset"]["metrics"][
        "train"
    ]
    epochs = range(1, len(metrics) + 1)

    plt.figure()
    plt.plot(epochs, metrics, label="Synthetic RAS Metric", color="orange")
    plt.title("Synthetic RAS Metrics for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("RAS Value")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_RASMetrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RAS metrics plot: {e}")
    plt.close()
