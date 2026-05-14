import matplotlib.pyplot as plt
import numpy as np
import os

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses for each feature set
for feature_set, key in zip(
    ["only_author_ratings", "only_review_scores", "both_features"],
    ["only_author_ratings", "only_review_scores", "both_features"],
):
    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][key]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["ablation_study"][key]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{key.capitalize().replace('_', ' ')} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {key}: {e}")
        plt.close()
