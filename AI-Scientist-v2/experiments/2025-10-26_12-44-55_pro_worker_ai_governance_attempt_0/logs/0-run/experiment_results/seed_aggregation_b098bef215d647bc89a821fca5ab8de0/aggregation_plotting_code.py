import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data_path_list = [
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_2845dfcba8ce4d43b56312c8b090eec4_proc_2527865/experiment_data.npy",
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_8b66699e29d1448da884f2c0c4a72241_proc_2527866/experiment_data.npy",
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_5ea869cd4abe4f0fac2949a885be72a7_proc_2527864/experiment_data.npy",
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

# Plot training and validation losses
try:
    losses = []
    for experiment_data in all_experiment_data:
        losses.append(
            experiment_data["hyperparam_tuning_learning_rate"]["synthetic_worker_data"][
                "losses"
            ]
        )
    losses_mean = {
        "train": np.mean([loss["train"] for loss in losses], axis=0),
        "val": np.mean([loss["val"] for loss in losses], axis=0),
    }
    losses_sem = {
        "train": np.std([loss["train"] for loss in losses], axis=0)
        / np.sqrt(len(losses)),
        "val": np.std([loss["val"] for loss in losses], axis=0) / np.sqrt(len(losses)),
    }
    epochs = range(len(losses_mean["train"]))

    plt.figure()
    plt.plot(epochs, losses_mean["train"], label="Mean Training Loss")
    plt.fill_between(
        epochs,
        losses_mean["train"] - losses_sem["train"],
        losses_mean["train"] + losses_sem["train"],
        alpha=0.3,
    )
    plt.plot(epochs, losses_mean["val"], label="Mean Validation Loss")
    plt.fill_between(
        epochs,
        losses_mean["val"] - losses_sem["val"],
        losses_mean["val"] + losses_sem["val"],
        alpha=0.3,
    )
    plt.title("Mean Training and Validation Losses with SEM")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_mean_training_validation_losses_sem.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot predictions vs ground truth
try:
    ground_truth = []
    predictions = []
    for experiment_data in all_experiment_data:
        gt = experiment_data["hyperparam_tuning_learning_rate"][
            "synthetic_worker_data"
        ]["ground_truth"]
        pred = experiment_data["hyperparam_tuning_learning_rate"][
            "synthetic_worker_data"
        ]["predictions"]
        ground_truth.append(gt)
        predictions.append(pred)
    ground_truth_mean = np.mean(ground_truth, axis=0)
    predictions_mean = np.mean(predictions, axis=0)
    predictions_sem = np.std(predictions, axis=0) / np.sqrt(len(predictions))

    plt.figure()
    plt.scatter(ground_truth_mean, predictions_mean, alpha=0.5)
    plt.errorbar(
        ground_truth_mean,
        predictions_mean,
        yerr=predictions_sem,
        fmt="o",
        label="Predictions with SEM",
        alpha=0.3,
    )
    plt.plot(
        [min(ground_truth_mean), max(ground_truth_mean)],
        [min(ground_truth_mean), max(ground_truth_mean)],
        color="red",
        linestyle="--",
    )
    plt.title("Predictions vs Ground Truth with SEM")
    plt.xlabel("Ground Truth WIS")
    plt.ylabel("Predicted WIS")
    plt.axis("equal")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_worker_data_predictions_vs_ground_truth_sem.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
