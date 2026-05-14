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

# Plot Training and Validation Loss
try:
    plt.figure()
    epochs = range(len(experiment_data["synthetic_dataset"]["losses"]["train"]))
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot Training and Validation Metrics
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["metrics"]["train"],
        label="Training Metric",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("Training and Validation Worker Impact Score (WIS)")
    plt.xlabel("Epochs")
    plt.ylabel("Worker Impact Score")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_wis.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating WIS plot: {e}")
    plt.close()
