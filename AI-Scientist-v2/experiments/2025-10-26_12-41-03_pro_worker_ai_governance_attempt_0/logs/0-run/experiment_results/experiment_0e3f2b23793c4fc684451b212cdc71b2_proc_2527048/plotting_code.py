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

# Plot Training and Validation Loss
try:
    plt.figure(figsize=(10, 5))
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["val"], label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

# Plot Ground Truth vs Predictions
try:
    all_predictions = np.array(experiment_data["synthetic_data"]["predictions"])
    all_ground_truth = np.array(experiment_data["synthetic_data"]["ground_truth"])
    plt.figure(figsize=(10, 5))
    plt.scatter(all_ground_truth, all_predictions, alpha=0.5)
    plt.xlabel("Ground Truth WIS")
    plt.ylabel("Predicted WIS")
    plt.title("Ground Truth vs Predicted Worker Impact Score (WIS)")
    plt.savefig(os.path.join(working_dir, "wis_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Ground Truth vs Predictions plot: {e}")
    plt.close()
