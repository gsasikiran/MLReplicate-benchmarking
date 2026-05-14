import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_65b5b32f575f48a58fbd4d23bd641dd0_proc_2531696/experiment_data.npy",
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_26f944ae9e8e45a59436f1ad80ceabe1_proc_2531698/experiment_data.npy",
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_61db22e4c6fb445ea8f05b56064c99df_proc_2531697/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)

    learning_rates = all_experiment_data[0]["hyperparam_tuning_learning_rate"][
        "synthetic_data"
    ]["learning_rates"]
    train_losses = [
        exp["hyperparam_tuning_learning_rate"]["synthetic_data"]["losses"]["train"]
        for exp in all_experiment_data
    ]
    val_losses = [
        exp["hyperparam_tuning_learning_rate"]["synthetic_data"]["losses"]["val"]
        for exp in all_experiment_data
    ]
    train_metrics = [
        exp["hyperparam_tuning_learning_rate"]["synthetic_data"]["metrics"]["train"]
        for exp in all_experiment_data
    ]
    val_metrics = [
        exp["hyperparam_tuning_learning_rate"]["synthetic_data"]["metrics"]["val"]
        for exp in all_experiment_data
    ]

    train_losses_mean = np.mean(train_losses, axis=0)
    val_losses_mean = np.mean(val_losses, axis=0)
    train_losses_error = np.std(train_losses, axis=0) / np.sqrt(
        len(all_experiment_data)
    )
    val_losses_error = np.std(val_losses, axis=0) / np.sqrt(len(all_experiment_data))

    try:
        plt.figure()
        for i, lr in enumerate(learning_rates):
            plt.plot(train_losses_mean[i], label=f"Train Loss (lr={lr})")
            plt.fill_between(
                np.arange(len(train_losses_mean[i])),
                train_losses_mean[i] - train_losses_error[i],
                train_losses_mean[i] + train_losses_error[i],
                alpha=0.2,
            )
            plt.plot(
                val_losses_mean[i], label=f"Validation Loss (lr={lr})", linestyle="--"
            )
            plt.fill_between(
                np.arange(len(val_losses_mean[i])),
                val_losses_mean[i] - val_losses_error[i],
                val_losses_mean[i] + val_losses_error[i],
                color="orange",
                alpha=0.2,
            )
        plt.title("Mean Loss Curves with Error Bars for Different Learning Rates")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "mean_loss_curves_with_error.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    train_metrics_mean = np.mean(train_metrics, axis=0)
    val_metrics_mean = np.mean(val_metrics, axis=0)
    train_metrics_error = np.std(train_metrics, axis=0) / np.sqrt(
        len(all_experiment_data)
    )
    val_metrics_error = np.std(val_metrics, axis=0) / np.sqrt(len(all_experiment_data))

    try:
        plt.figure()
        for i, lr in enumerate(learning_rates):
            plt.plot(train_metrics_mean[i], label=f"Train Accuracy (lr={lr})")
            plt.fill_between(
                np.arange(len(train_metrics_mean[i])),
                train_metrics_mean[i] - train_metrics_error[i],
                train_metrics_mean[i] + train_metrics_error[i],
                alpha=0.2,
            )
            plt.plot(
                val_metrics_mean[i],
                label=f"Validation Accuracy (lr={lr})",
                linestyle="--",
            )
            plt.fill_between(
                np.arange(len(val_metrics_mean[i])),
                val_metrics_mean[i] - val_metrics_error[i],
                val_metrics_mean[i] + val_metrics_error[i],
                color="orange",
                alpha=0.2,
            )
        plt.title("Mean Accuracy Curves with Error Bars for Different Learning Rates")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "mean_accuracy_curves_with_error.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves plot: {e}")
        plt.close()

except Exception as e:
    print(f"Error loading experiment data: {e}")
