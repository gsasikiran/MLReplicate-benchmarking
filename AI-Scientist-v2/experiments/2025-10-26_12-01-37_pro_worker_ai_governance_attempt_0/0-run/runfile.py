import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_815425a113524f178e442350de5df3f8_proc_2526421/experiment_data.npy",
    "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_36edc40ca51949aa9db52d22c8a45826_proc_2526419/experiment_data.npy",
    "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_2bfda171cf3d412abb0ddca6243a3575_proc_2526418/experiment_data.npy",
]

all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data: {e}")

for dataset_name in all_experiment_data[0]["ablation_study"].keys():
    losses_train = []
    losses_val = []

    # Aggregating losses across experiments for the dataset
    for experiment_data in all_experiment_data:
        losses_train.append(
            experiment_data["ablation_study"][dataset_name]["losses"]["train"]
        )
        losses_val.append(
            experiment_data["ablation_study"][dataset_name]["losses"]["val"]
        )

    losses_train_mean = np.mean(losses_train, axis=0)
    losses_train_sem = np.std(losses_train, axis=0) / np.sqrt(len(losses_train))
    losses_val_mean = np.mean(losses_val, axis=0)
    losses_val_sem = np.std(losses_val, axis=0) / np.sqrt(len(losses_val))

    try:
        plt.figure()
        epochs = np.arange(len(losses_train_mean))
        plt.plot(epochs, losses_train_mean, label="Training Loss")
        plt.fill_between(
            epochs,
            losses_train_mean - losses_train_sem,
            losses_train_mean + losses_train_sem,
            alpha=0.2,
        )
        plt.plot(epochs, losses_val_mean, label="Validation Loss")
        plt.fill_between(
            epochs,
            losses_val_mean - losses_val_sem,
            losses_val_mean + losses_val_sem,
            alpha=0.2,
        )
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_loss_curves_aggregated.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curves plot: {e}")
        plt.close()

    try:
        gt = []
        preds = []
        for experiment_data in all_experiment_data:
            gt.extend(experiment_data["ablation_study"][dataset_name]["ground_truth"])
            preds.extend(experiment_data["ablation_study"][dataset_name]["predictions"])

        plt.figure()
        plt.scatter(gt, preds, alpha=0.5)
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")
        plt.title(f"{dataset_name} Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(
                working_dir,
                f"{dataset_name}_predictions_vs_ground_truth_aggregated.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions scatter plot: {e}")
        plt.close()
