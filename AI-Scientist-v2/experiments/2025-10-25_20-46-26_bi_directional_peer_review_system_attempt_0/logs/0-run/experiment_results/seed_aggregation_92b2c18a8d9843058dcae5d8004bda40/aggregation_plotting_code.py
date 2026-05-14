import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data_path_list = [
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_ea8be50513704e0685c96edd3f5f98d7_proc_2515358/experiment_data.npy",
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_5afced5afcdf4e939cc6229571d6688c_proc_2515359/experiment_data.npy",
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_1b3488d8d4db44d0bc2bfe7a76c1e06a_proc_2515361/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

for feature in range(4):
    try:
        plt.figure()
        losses = [
            data["feature_importance_removal"]["FeedbackDataset"]["losses"]["train"]
            for data in all_experiment_data
        ]
        mean_losses = np.mean(losses, axis=0)
        std_losses = np.std(losses, axis=0)
        epochs = np.arange(len(mean_losses))
        plt.plot(mean_losses, label=f"Mean Loss - Feature Removed: {feature}")
        plt.fill_between(
            epochs,
            mean_losses - std_losses,
            mean_losses + std_losses,
            alpha=0.3,
            label="Standard Error",
        )
        plt.title("Aggregated Training Loss Over Epochs - FeedbackDataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"aggregated_training_loss_feature_{feature}.png")
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating aggregated training loss plot for feature {feature}: {e}"
        )
        plt.close()

for feature in range(4):
    try:
        plt.figure()
        val_losses = [
            data["feature_importance_removal"]["FeedbackDataset"]["losses"]["val"]
            for data in all_experiment_data
        ]
        mean_val_losses = np.mean(val_losses, axis=0)
        std_val_losses = np.std(val_losses, axis=0)
        epochs = np.arange(len(mean_val_losses))
        plt.plot(
            mean_val_losses, label=f"Mean Validation Loss - Feature Removed: {feature}"
        )
        plt.fill_between(
            epochs,
            mean_val_losses - std_val_losses,
            mean_val_losses + std_val_losses,
            alpha=0.3,
            label="Standard Error",
        )
        plt.title("Aggregated Validation Loss Over Epochs - FeedbackDataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"aggregated_validation_loss_feature_{feature}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating aggregated validation loss plot for feature {feature}: {e}"
        )
        plt.close()
