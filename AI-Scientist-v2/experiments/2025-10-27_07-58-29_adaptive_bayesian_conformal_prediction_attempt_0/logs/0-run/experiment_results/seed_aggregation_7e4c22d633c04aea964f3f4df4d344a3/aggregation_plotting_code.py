import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_3a7997fc0e2346419748cff68d04968a_proc_2543963/experiment_data.npy",
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_a5b70756be874363a0d6c4a4760bc2e8_proc_2543960/experiment_data.npy",
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_33681a0a57d14893ab8f770183605601_proc_2543962/experiment_data.npy",
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

# Plot training and validation loss with mean and standard error
try:
    train_losses = [
        exp["hyperparam_tuning_momentum"]["synthetic_data"]["losses"]["train"]
        for exp in all_experiment_data
    ]
    val_losses = [
        exp["hyperparam_tuning_momentum"]["synthetic_data"]["losses"]["val"]
        for exp in all_experiment_data
    ]

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_val_losses = np.mean(val_losses, axis=0)
    std_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(all_experiment_data))
    std_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(all_experiment_data))

    plt.figure()
    plt.plot(mean_train_losses, label="Mean Training Loss")
    plt.fill_between(
        range(len(mean_train_losses)),
        mean_train_losses - std_train_losses,
        mean_train_losses + std_train_losses,
        alpha=0.3,
    )

    plt.plot(mean_val_losses, label="Mean Validation Loss")
    plt.fill_between(
        range(len(mean_val_losses)),
        mean_val_losses - std_val_losses,
        mean_val_losses + std_val_losses,
        alpha=0.3,
    )

    plt.title("Mean Loss Curves for Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_mean_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss curve plot: {e}")
    plt.close()

# Plot predictions vs ground truth with mean and standard error
try:
    predictions_all = [
        exp["hyperparam_tuning_momentum"]["synthetic_data"]["predictions"]
        for exp in all_experiment_data
    ]
    ground_truth_all = [
        exp["hyperparam_tuning_momentum"]["synthetic_data"]["ground_truth"]
        for exp in all_experiment_data
    ]

    mean_predictions = np.mean(predictions_all, axis=0)
    mean_ground_truth = np.mean(ground_truth_all, axis=0)

    std_predictions = np.std(predictions_all, axis=0)
    std_ground_truth = np.std(ground_truth_all, axis=0)

    plt.figure()
    plt.scatter(mean_ground_truth, mean_predictions)
    plt.errorbar(
        mean_ground_truth,
        mean_predictions,
        xerr=std_ground_truth,
        yerr=std_predictions,
        fmt="o",
        label="Mean ± Std Err",
        alpha=0.5,
    )
    plt.plot(
        [mean_ground_truth.min(), mean_ground_truth.max()],
        [mean_ground_truth.min(), mean_ground_truth.max()],
        "r--",
    )
    plt.title("Mean Predictions vs Ground Truth for Synthetic Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Model Predictions")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_mean_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean predictions vs ground truth plot: {e}")
    plt.close()
