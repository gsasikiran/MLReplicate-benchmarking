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

# Plotting training loss
try:
    plt.figure()
    epochs = np.arange(
        len(
            experiment_data["hyperparam_tuning_optimizer"]["peer_review_feedback"][
                "losses"
            ]["train"]
        )
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_optimizer"]["peer_review_feedback"][
            "losses"
        ]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_optimizer"]["peer_review_feedback"][
            "losses"
        ]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_feedback_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plotting metrics
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_optimizer"]["peer_review_feedback"][
            "metrics"
        ]["train"],
        label="Training Metrics",
    )
    plt.title("Training Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_feedback_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
