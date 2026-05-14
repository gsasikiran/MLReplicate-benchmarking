import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_paths = [
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_bf8b8026a8df47beb4803fd975886830_proc_2529891/experiment_data.npy",
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_bb4b8d9c4e614f45bbde6c116ef9ddc8_proc_2529890/experiment_data.npy",
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_be856ba0d6054054a8991712a4eabb09_proc_2529893/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate and plot training and validation loss for different weight decays
for wd in all_experiment_data[0]["weight_decay_tuning"]:
    try:
        train_losses = []
        val_losses = []
        for exp in all_experiment_data:
            train_losses.append(exp["weight_decay_tuning"][wd]["losses"]["train"])
            val_losses.append(exp["weight_decay_tuning"][wd]["losses"]["val"])

        # Calculate mean and standard error
        mean_train_losses = np.mean(train_losses, axis=0)
        mean_val_losses = np.mean(val_losses, axis=0)
        se_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        se_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))
        epochs = range(1, len(mean_train_losses) + 1)

        plt.figure()
        plt.plot(epochs, mean_train_losses, label="Mean Train Loss")
        plt.fill_between(
            epochs,
            mean_train_losses - se_train_losses,
            mean_train_losses + se_train_losses,
            alpha=0.2,
            label="SE Train Loss",
        )
        plt.plot(epochs, mean_val_losses, label="Mean Validation Loss")
        plt.fill_between(
            epochs,
            mean_val_losses - se_val_losses,
            mean_val_losses + se_val_losses,
            alpha=0.2,
            label="SE Validation Loss",
        )
        plt.title(f"Aggregated Loss Curves for Weight Decay: {wd}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"aggregated_loss_curves_wd_{wd}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for weight decay {wd}: {e}")

# Aggregate predictions vs ground truth plots can follow a similar approach if needed.
