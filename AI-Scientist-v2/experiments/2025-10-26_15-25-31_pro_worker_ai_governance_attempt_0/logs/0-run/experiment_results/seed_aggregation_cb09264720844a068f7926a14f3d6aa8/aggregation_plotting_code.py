import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_793a40eeac8443bdbe19c6cc50c9c5e3_proc_2531283/experiment_data.npy",
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_ade5298285f4463ebb1ea0400c3e3cde_proc_2531284/experiment_data.npy",
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_dfb466a48f3a439690e62611ef1f686e_proc_2531282/experiment_data.npy",
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

learning_rates = [
    exp["hyperparam_tuning_learning_rate"]["synthetic_data"]["learning_rates"]
    for exp in all_experiment_data
]
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

try:
    # Loss Plot
    plt.figure()
    mean_train_losses = np.mean(train_losses, axis=0)
    mean_val_losses = np.mean(val_losses, axis=0)
    se_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(all_experiment_data))
    se_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(all_experiment_data))

    plt.plot(mean_train_losses, label="Mean Train Loss")
    plt.fill_between(
        range(len(mean_train_losses)),
        mean_train_losses - se_train_losses,
        mean_train_losses + se_train_losses,
        alpha=0.2,
    )
    plt.plot(mean_val_losses, label="Mean Val Loss", linestyle="--")
    plt.fill_between(
        range(len(mean_val_losses)),
        mean_val_losses - se_val_losses,
        mean_val_losses + se_val_losses,
        alpha=0.2,
    )

    plt.title("Mean Loss Curves with Standard Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss plot: {e}")
    plt.close()

try:
    # Metric Plot
    plt.figure()
    mean_train_metrics = np.mean(train_metrics, axis=0)
    mean_val_metrics = np.mean(val_metrics, axis=0)
    se_train_metrics = np.std(train_metrics, axis=0) / np.sqrt(len(all_experiment_data))
    se_val_metrics = np.std(val_metrics, axis=0) / np.sqrt(len(all_experiment_data))

    plt.plot(mean_train_metrics, label="Mean Train Accuracy")
    plt.fill_between(
        range(len(mean_train_metrics)),
        mean_train_metrics - se_train_metrics,
        mean_train_metrics + se_train_metrics,
        alpha=0.2,
    )
    plt.plot(mean_val_metrics, label="Mean Val Accuracy", linestyle="--")
    plt.fill_between(
        range(len(mean_val_metrics)),
        mean_val_metrics - se_val_metrics,
        mean_val_metrics + se_val_metrics,
        alpha=0.2,
    )

    plt.title("Mean Accuracy Curves with Standard Error")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean accuracy plot: {e}")
    plt.close()
