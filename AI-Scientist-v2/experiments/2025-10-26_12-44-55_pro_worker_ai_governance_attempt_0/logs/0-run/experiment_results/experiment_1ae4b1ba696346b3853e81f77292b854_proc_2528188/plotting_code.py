import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses
for noise_type in ["no_noise", "low_noise", "medium_noise", "high_noise"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["data_noise_impact"][noise_type]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["data_noise_impact"][noise_type]["losses"]["val"],
            label="Val Loss",
        )
        plt.title(f'Losses for {noise_type.replace("_", " ").title()} Dataset')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{working_dir}/{noise_type}_losses.png")
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {noise_type} losses: {e}")
        plt.close()

# Plot predictions vs ground truths
for noise_type in ["no_noise", "low_noise", "medium_noise", "high_noise"]:
    try:
        plt.figure()
        plt.scatter(
            experiment_data["data_noise_impact"][noise_type]["ground_truth"],
            experiment_data["data_noise_impact"][noise_type]["predictions"],
            alpha=0.5,
        )
        plt.title(
            f'Predictions vs Ground Truth for {noise_type.replace("_", " ").title()} Dataset'
        )
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.axis("equal")
        plt.plot(
            [
                min(experiment_data["data_noise_impact"][noise_type]["ground_truth"]),
                max(experiment_data["data_noise_impact"][noise_type]["ground_truth"]),
            ],
            [
                min(experiment_data["data_noise_impact"][noise_type]["ground_truth"]),
                max(experiment_data["data_noise_impact"][noise_type]["ground_truth"]),
            ],
            "r--",
        )  # 45-degree line
        plt.savefig(f"{working_dir}/{noise_type}_predictions_vs_ground_truth.png")
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {noise_type} predictions vs ground truth: {e}")
        plt.close()
