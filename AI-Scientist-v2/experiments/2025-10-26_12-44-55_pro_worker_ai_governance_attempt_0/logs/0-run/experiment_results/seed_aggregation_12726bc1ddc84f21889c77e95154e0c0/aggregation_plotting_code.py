import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data_paths = [
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_bb2a2ace271b4ad7b23d8a5f8ae21d6d_proc_2527451/experiment_data.npy",
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_0fc4de3e36c64817af58093cd9aac796_proc_2527448/experiment_data.npy",
        "experiments/2025-10-26_12-44-55_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_c870aadd61cc41c08f7e35d630708869_proc_2527449/experiment_data.npy",
    ]
    all_experiment_data = []
    for path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot aggregated training and validation losses
try:
    all_losses = [
        data["hyperparam_tuning_learning_rate"]["synthetic_worker_data"]["losses"]
        for data in all_experiment_data
    ]
    train_losses = np.array([loss["train"] for loss in all_losses])
    val_losses = np.array([loss["val"] for loss in all_losses])
    epochs = range(train_losses.shape[1])  # Assuming all training lengths are equal

    mean_train_losses = train_losses.mean(axis=0)
    mean_val_losses = val_losses.mean(axis=0)
    ste_train_losses = train_losses.std(axis=0) / np.sqrt(train_losses.shape[0])
    ste_val_losses = val_losses.std(axis=0) / np.sqrt(val_losses.shape[0])

    plt.figure()
    plt.plot(epochs, mean_train_losses, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_train_losses - ste_train_losses,
        mean_train_losses + ste_train_losses,
        alpha=0.2,
    )
    plt.plot(epochs, mean_val_losses, label="Mean Validation Loss")
    plt.fill_between(
        epochs,
        mean_val_losses - ste_val_losses,
        mean_val_losses + ste_val_losses,
        alpha=0.2,
    )
    plt.title("Aggregated Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir,
            "synthetic_worker_data_aggregated_training_validation_losses.png",
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# Plot aggregated predictions vs ground truth
try:
    predictions = np.concatenate(
        [
            data["hyperparam_tuning_learning_rate"]["synthetic_worker_data"][
                "predictions"
            ]
            for data in all_experiment_data
        ]
    )
    ground_truth = np.concatenate(
        [
            data["hyperparam_tuning_learning_rate"]["synthetic_worker_data"][
                "ground_truth"
            ]
            for data in all_experiment_data
        ]
    )

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        color="red",
        linestyle="--",
    )
    plt.title("Aggregated Predictions vs Ground Truth")
    plt.xlabel("Ground Truth WIS")
    plt.ylabel("Predicted WIS")
    plt.axis("equal")
    plt.savefig(
        os.path.join(
            working_dir,
            "synthetic_worker_data_aggregated_predictions_vs_ground_truth.png",
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated predictions plot: {e}")
    plt.close()
