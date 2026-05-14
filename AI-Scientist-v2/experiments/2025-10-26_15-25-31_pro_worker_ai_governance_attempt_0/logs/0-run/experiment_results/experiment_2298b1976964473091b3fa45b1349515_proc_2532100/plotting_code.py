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

for feature_count in experiment_data["feature_count_ablation"]["synthetic_data"][
    "feature_counts"
]:
    try:
        losses_train = experiment_data["feature_count_ablation"]["synthetic_data"][
            "losses"
        ]["train"]
        losses_val = experiment_data["feature_count_ablation"]["synthetic_data"][
            "losses"
        ]["val"]
        plt.figure()
        plt.plot(losses_train, label="Training Loss")
        plt.plot(losses_val, label="Validation Loss")
        plt.title(f"Losses with {feature_count} Features")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"losses_feature_count_{feature_count}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for losses with feature count {feature_count}: {e}")
        plt.close()

    try:
        metrics_train = experiment_data["feature_count_ablation"]["synthetic_data"][
            "metrics"
        ]["train"]
        metrics_val = experiment_data["feature_count_ablation"]["synthetic_data"][
            "metrics"
        ]["val"]
        plt.figure()
        plt.plot(metrics_train, label="Training Score")
        plt.plot(metrics_val, label="Validation Score")
        plt.title(f"Metrics with {feature_count} Features")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"metrics_feature_count_{feature_count}.png")
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating plot for metrics with feature count {feature_count}: {e}"
        )
        plt.close()
