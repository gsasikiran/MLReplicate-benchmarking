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

# Plotting training and validation losses
try:
    epochs = range(
        len(experiment_data["early_stopping"]["synthetic_data"]["losses"]["train"])
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["early_stopping"]["synthetic_data"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["early_stopping"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plotting validation predictions vs ground truth
try:
    val_predictions = experiment_data["early_stopping"]["synthetic_data"]["predictions"]
    val_ground_truth = experiment_data["early_stopping"]["synthetic_data"][
        "ground_truth"
    ]
    for epoch in range(0, len(val_predictions), max(1, len(val_predictions) // 5)):
        plt.figure()
        plt.scatter(
            val_ground_truth[epoch],
            val_predictions[epoch],
            label="Predictions",
            color="blue",
        )
        plt.plot(
            [min(val_ground_truth[epoch]), max(val_ground_truth[epoch])],
            [min(val_ground_truth[epoch]), max(val_ground_truth[epoch])],
            color="red",
            linestyle="--",
        )
        plt.title(f"Validation Predictions vs Ground Truth at Epoch {epoch}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"synthetic_data_validation_predictions_epoch_{epoch}.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
