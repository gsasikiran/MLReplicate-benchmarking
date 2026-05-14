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
    for wd in experiment_data["feature_interaction_ablation"]["original_features"]:
        plt.figure()
        plt.plot(
            experiment_data["feature_interaction_ablation"]["original_features"][wd][
                "losses"
            ]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["feature_interaction_ablation"]["original_features"][wd][
                "losses"
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"Original Features - Weight Decay {wd}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"original_features_loss_wd_{wd}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plot for original features: {e}")
    plt.close()

try:
    for wd in experiment_data["feature_interaction_ablation"]["interaction_features"]:
        plt.figure()
        plt.plot(
            experiment_data["feature_interaction_ablation"]["interaction_features"][wd][
                "losses"
            ]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["feature_interaction_ablation"]["interaction_features"][wd][
                "losses"
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"Interaction Features - Weight Decay {wd}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"interaction_features_loss_wd_{wd}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plot for interaction features: {e}")
    plt.close()
