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

try:
    plt.figure()
    plt.plot(
        experiment_data["peer_review_experiment"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "peer_review_experiment_training_loss_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating Training Loss Plot: {e}")
    plt.close()

try:
    if "val" in experiment_data["peer_review_experiment"]["losses"]:
        plt.figure()
        plt.plot(
            experiment_data["peer_review_experiment"]["losses"]["val"],
            label="Validation Loss",
            color="orange",
        )
        plt.title("Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, "peer_review_experiment_validation_loss_curve.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating Validation Loss Plot: {e}")
    plt.close()

# Potentially add other plots like prediction vs ground truth if present
try:
    plt.figure()
    plt.scatter(
        experiment_data["peer_review_experiment"]["ground_truth"],
        experiment_data["peer_review_experiment"]["predictions"],
        alpha=0.5,
    )
    plt.title("Ground Truth vs Predictions")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(
            working_dir, "peer_review_experiment_ground_truth_vs_predictions.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating Ground Truth vs Predictions Plot: {e}")
    plt.close()
