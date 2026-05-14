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

# Plotting training and validation loss curves
for config in experiment_data["feature_influence_ablation"]:
    try:
        metrics = experiment_data["feature_influence_ablation"][config]["metrics"]
        losses = experiment_data["feature_influence_ablation"][config]["losses"]

        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"{config} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{config}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {config}: {e}")
        plt.close()

# Plotting ground truth vs predictions for the full config
try:
    truth = np.concatenate(
        experiment_data["feature_influence_ablation"]["full"]["ground_truth"]
    )
    preds = np.concatenate(
        experiment_data["feature_influence_ablation"]["full"]["predictions"]
    )

    plt.figure()
    plt.scatter(truth, preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], "--", color="red")
    plt.title("Ground Truth vs Predictions (Full)")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "full_ground_truth_vs_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
