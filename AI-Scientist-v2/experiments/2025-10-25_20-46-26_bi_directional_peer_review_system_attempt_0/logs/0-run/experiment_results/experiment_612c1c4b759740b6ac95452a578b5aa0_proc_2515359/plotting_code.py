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

# Training Loss Plot
try:
    plt.figure()
    epochs = range(
        1,
        len(
            experiment_data["batch_normalization"]["FeedbackDataset"]["losses"]["train"]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["batch_normalization"]["FeedbackDataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["batch_normalization"]["FeedbackDataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "FeedbackDataset_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating Training/Validation Loss plot: {e}")
    plt.close()

# Metrics Plot (Synthetic RAS)
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["batch_normalization"]["FeedbackDataset"]["metrics"]["train"],
        label="Train RAS Metric",
    )
    plt.title("Training Metrics for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("Synthetic RAS Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_training_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training Metrics plot: {e}")
    plt.close()
