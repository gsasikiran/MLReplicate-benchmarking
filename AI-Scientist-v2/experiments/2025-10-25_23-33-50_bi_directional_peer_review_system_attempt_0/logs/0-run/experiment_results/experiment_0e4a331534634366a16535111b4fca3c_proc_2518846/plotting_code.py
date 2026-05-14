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

for feature in ["only_author_ratings", "only_review_scores", "both_features"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][feature]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["ablation_study"][feature]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{feature} - Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{feature}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {feature} losses: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(
            experiment_data["ablation_study"][feature]["ground_truth"],
            experiment_data["ablation_study"][feature]["predictions"],
        )
        plt.title(f"{feature} - Ground Truth vs Predictions")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.plot([0, 1], [0, 1], linestyle="--", color="red")  # y=x line
        plt.savefig(
            os.path.join(working_dir, f"{feature}_ground_truth_vs_predictions.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {feature} ground truth vs predictions: {e}")
        plt.close()
