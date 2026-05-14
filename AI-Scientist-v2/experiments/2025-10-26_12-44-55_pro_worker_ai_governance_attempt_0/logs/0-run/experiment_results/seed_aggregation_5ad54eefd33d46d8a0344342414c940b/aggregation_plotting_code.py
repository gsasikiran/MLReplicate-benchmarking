import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_17ca7fa3f56e452fbc27e2059cf4f4e7_proc_2527261/experiment_data.npy",
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_cdda8e4c48844268ac8e14d88728074a_proc_2527262/experiment_data.npy",
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_c461bdc03c5f4d4dafb982df5081f64f_proc_2527260/experiment_data.npy",
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

try:
    plt.figure()
    train_losses = np.array(
        [exp["synthetic_worker_data"]["losses"]["train"] for exp in all_experiment_data]
    )
    val_losses = np.array(
        [exp["synthetic_worker_data"]["losses"]["val"] for exp in all_experiment_data]
    )
    epochs = np.arange(1, train_losses.shape[1] + 1)

    train_mean = train_losses.mean(axis=0)
    val_mean = val_losses.mean(axis=0)
    train_se = train_losses.std(axis=0) / np.sqrt(len(all_experiment_data))
    val_se = val_losses.std(axis=0) / np.sqrt(len(all_experiment_data))

    plt.plot(epochs, train_mean, label="Training Loss")
    plt.fill_between(epochs, train_mean - train_se, train_mean + train_se, alpha=0.1)
    plt.plot(epochs, val_mean, label="Validation Loss")
    plt.fill_between(epochs, val_mean - val_se, val_mean + val_se, alpha=0.1)

    plt.title("Mean Training and Validation Losses with SE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worker_data_losses_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    ground_truths = np.array(
        [exp["synthetic_worker_data"]["ground_truth"] for exp in all_experiment_data]
    )
    predictions = np.array(
        [exp["synthetic_worker_data"]["predictions"] for exp in all_experiment_data]
    )

    mean_gt = ground_truths.mean(axis=0)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    plt.scatter(mean_gt, mean_pred, alpha=0.5)
    plt.errorbar(
        mean_gt, mean_pred, yerr=std_pred, fmt="o", alpha=0.5, label="Prediction Error"
    )
    plt.plot([0, 1], [0, 1], "r--")  # Ideal line
    plt.title("Predictions vs Ground Truth with Error Bars")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_predictions_vs_truth_errorbars.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
