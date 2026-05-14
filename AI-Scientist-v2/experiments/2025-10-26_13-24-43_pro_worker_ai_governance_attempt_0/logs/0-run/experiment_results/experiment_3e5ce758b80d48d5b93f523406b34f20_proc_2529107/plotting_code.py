import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    epochs = range(1, len(experiment_data["synthetic_dataset"]["losses"]["train"]) + 1)
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot for training and validation loss: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs, experiment_data["synthetic_dataset"]["metrics"]["train"], label="PWIS"
    )
    plt.title("Pro-Worker Impact Score Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_pwis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for PWIS: {e}")
    plt.close()

try:
    plt.figure()
    plt.scatter(
        experiment_data["synthetic_dataset"]["ground_truth"],
        experiment_data["synthetic_dataset"]["predictions"],
        alpha=0.5,
    )
    plt.title("Ground Truth vs Predictions")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.axhline(0.5, color="red", linestyle="--")
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_ground_truth_vs_predictions.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
