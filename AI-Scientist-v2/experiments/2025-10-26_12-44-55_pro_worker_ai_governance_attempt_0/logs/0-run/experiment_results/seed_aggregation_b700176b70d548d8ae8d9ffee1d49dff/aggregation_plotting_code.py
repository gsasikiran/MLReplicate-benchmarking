import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_9c38363130674e06991806c3c83a68e3_proc_2528187/experiment_data.npy",
    "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_8a1d79404855444f89616d915f7932ac_proc_2528188/experiment_data.npy",
]

try:
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Assuming the metrics have a consistent structure
try:
    plt.figure()
    epochs = np.arange(
        1,
        len(
            all_experiment_data[0]["loss_function_ablation"]["synthetic_worker_data"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    train_losses = [
        data["loss_function_ablation"]["synthetic_worker_data"]["losses"]["train"]
        for data in all_experiment_data
    ]
    val_losses = [
        data["loss_function_ablation"]["synthetic_worker_data"]["losses"]["val"]
        for data in all_experiment_data
    ]

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_val_losses = np.mean(val_losses, axis=0)
    se_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
    se_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    plt.plot(epochs, mean_train_losses, label="Mean Train Loss")
    plt.fill_between(
        epochs,
        mean_train_losses - se_train_losses,
        mean_train_losses + se_train_losses,
        alpha=0.1,
    )
    plt.plot(epochs, mean_val_losses, label="Mean Validation Loss")
    plt.fill_between(
        epochs,
        mean_val_losses - se_val_losses,
        mean_val_losses + se_val_losses,
        alpha=0.1,
    )
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_worker_data_loss_curve_mean_se.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    plt.figure()
    val_metrics = [
        data["loss_function_ablation"]["synthetic_worker_data"]["metrics"]["val"]
        for data in all_experiment_data
    ]

    mean_val_metrics = np.mean(val_metrics, axis=0)
    se_val_metrics = np.std(val_metrics, axis=0) / np.sqrt(len(val_metrics))

    plt.plot(epochs, mean_val_metrics, label="Mean WIS")
    plt.fill_between(
        epochs,
        mean_val_metrics - se_val_metrics,
        mean_val_metrics + se_val_metrics,
        alpha=0.1,
    )
    plt.title("Validation Metric (WIS) Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Worker Impact Score (WIS)")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_worker_data_wis_curve_mean_se.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating WIS plot: {e}")
    plt.close()

try:
    predictions = np.concatenate(
        [
            data["loss_function_ablation"]["synthetic_worker_data"]["predictions"]
            for data in all_experiment_data
        ]
    )
    ground_truth = np.concatenate(
        [
            data["loss_function_ablation"]["synthetic_worker_data"]["ground_truth"]
            for data in all_experiment_data
        ]
    )
    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Validation Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.axline((0, 0), slope=1, color="r", linestyle="--")  # 45-degree reference line
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
