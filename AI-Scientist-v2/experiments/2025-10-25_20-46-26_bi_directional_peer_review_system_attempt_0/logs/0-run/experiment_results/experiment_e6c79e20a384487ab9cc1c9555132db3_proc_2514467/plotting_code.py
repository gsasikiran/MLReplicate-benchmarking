import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Plot training and validation losses
    epochs = range(
        1,
        len(
            experiment_data["weight_decay_tuning"]["feedback_dataset"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["weight_decay_tuning"]["feedback_dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["weight_decay_tuning"]["feedback_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss over Epochs (Feedback Dataset)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "feedback_dataset_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot training metrics
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["weight_decay_tuning"]["feedback_dataset"]["metrics"]["train"],
        label="Training Metric",
    )
    plt.title("Training Metric over Epochs (Feedback Dataset)")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "feedback_dataset_metric_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()
