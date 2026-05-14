import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for dataset_name in experiment_data["input_feature_correlation_ablation"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["input_feature_correlation_ablation"][dataset_name][
                "losses"
            ]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["input_feature_correlation_ablation"][dataset_name][
                "losses"
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dataset_name}: {e}")
        plt.close()

    try:
        predictions = experiment_data["input_feature_correlation_ablation"][
            dataset_name
        ]["predictions"]
        ground_truth = experiment_data["input_feature_correlation_ablation"][
            dataset_name
        ]["ground_truth"]
        plt.figure()
        plt.scatter(
            ground_truth, predictions, label="Predictions vs Ground Truth", alpha=0.5
        )
        plt.plot([0, 1], [0, 1], "r--")  # Line of perfect prediction
        plt.title(f"{dataset_name} Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {dataset_name}: {e}")
        plt.close()
