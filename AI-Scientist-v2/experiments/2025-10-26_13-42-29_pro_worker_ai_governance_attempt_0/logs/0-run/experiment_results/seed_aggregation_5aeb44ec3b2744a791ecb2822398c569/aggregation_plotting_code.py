import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_cd61071931ed405dad54201b5a49d378_proc_2529599/experiment_data.npy",
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_2c3eae6228514f4e9038bb295513f2c2_proc_2529602/experiment_data.npy",
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_8586ebac71904e3b8dca5a552a36f4b1_proc_2529601/experiment_data.npy",
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

for wd in experiment_data["weight_decay_tuning"]:
    train_losses = []
    val_losses = []

    for data in all_experiment_data:
        train_losses.append(data["weight_decay_tuning"][wd]["losses"]["train"])
        val_losses.append(data["weight_decay_tuning"][wd]["losses"]["val"])

    # Calculate mean and standard error
    train_losses_mean = np.mean(train_losses, axis=0)
    val_losses_mean = np.mean(val_losses, axis=0)
    train_losses_se = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
    val_losses_se = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    try:
        epochs = range(1, len(train_losses_mean) + 1)
        plt.figure()
        plt.plot(epochs, train_losses_mean, label="Mean Train Loss")
        plt.fill_between(
            epochs,
            train_losses_mean - train_losses_se,
            train_losses_mean + train_losses_se,
            alpha=0.2,
        )
        plt.plot(epochs, val_losses_mean, label="Mean Validation Loss")
        plt.fill_between(
            epochs,
            val_losses_mean - val_losses_se,
            val_losses_mean + val_losses_se,
            alpha=0.2,
        )
        plt.title(f"Mean Loss Curves for Weight Decay: {wd}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"mean_loss_curves_wd_{wd}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating mean loss plot for weight decay {wd}: {e}")

    try:
        plt.figure()
        plt.scatter(
            data["weight_decay_tuning"][wd]["ground_truth"],
            data["weight_decay_tuning"][wd]["predictions"],
            alpha=0.5,
        )
        plt.plot([0, 1], [0, 1], "r--")  # Diagonal line
        plt.title(f"Predictions vs Ground Truth for Weight Decay: {wd}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"predictions_vs_ground_truth_wd_{wd}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for weight decay {wd}: {e}")
