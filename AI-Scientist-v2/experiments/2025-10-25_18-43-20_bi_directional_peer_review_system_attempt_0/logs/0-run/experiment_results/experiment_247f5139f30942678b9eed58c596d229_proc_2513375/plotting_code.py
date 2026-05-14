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

# Plot training and validation losses
try:
    plt.figure()
    epochs = range(len(experiment_data["rewards_system"]["losses"]["train"]))
    plt.plot(
        epochs,
        experiment_data["rewards_system"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["rewards_system"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses for Rewards System")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "rewards_system_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training and validation loss plot: {e}")
    plt.close()

# Plot ground truth vs predictions if available
try:
    if (
        experiment_data["rewards_system"]["ground_truth"]
        and experiment_data["rewards_system"]["predictions"]
    ):
        plt.figure()
        plt.scatter(
            experiment_data["rewards_system"]["ground_truth"],
            experiment_data["rewards_system"]["predictions"],
        )
        plt.title("Ground Truth vs Predictions for Rewards System")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, "rewards_system_ground_truth_vs_predictions.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()

# Additional plots for generated samples if applicable
try:
    if len(experiment_data["rewards_system"]["ris_scores"]) > 0:
        plt.figure()
        plt.plot(experiment_data["rewards_system"]["ris_scores"])
        plt.title("RIS Scores Over Epochs for Rewards System")
        plt.xlabel("Epochs")
        plt.ylabel("RIS Score")
        plt.savefig(os.path.join(working_dir, "rewards_system_ris_scores.png"))
        plt.close()
except Exception as e:
    print(f"Error creating RIS scores plot: {e}")
    plt.close()
