markdown
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load multiple experiment data files
experiment_data_path_list = [
    "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_d20c4ae38a2a40389a6661eec317a5c2_proc_2543541/experiment_data.npy",
    "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_44aa8458613b449f8b365e35212c18a8_proc_2543539/experiment_data.npy",
    "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_6633f59a722545c480a5b23d40e3d7ab_proc_2543540/experiment_data.npy",
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

# Aggregate and plot Loss Curves
try:
    train_losses = []
    val_losses = []
    for experiment_data in all_experiment_data:
        train_losses.append(
            experiment_data["hyperparam_tuning_momentum"]["synthetic_data"]["losses"][
                "train"
            ]
        )
        val_losses.append(
            experiment_data["hyperparam_tuning_momentum"]["synthetic_data"]["losses"][
                "val"
            ]
        )

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    train_mean = train_losses.mean(axis=0)
    val_mean = val_losses.mean(axis=0)
    train_se = train_losses.std(axis=0) / np.sqrt(len(all_experiment_data))
    val_se = val_losses.std(axis=0) / np.sqrt(len(all_experiment_data))

    plt.figure()
    plt.plot(train_mean, label="Training Loss")
    plt.fill_between(
        range(len(train_mean)), train_mean - train_se, train_mean + train_se, alpha=0.2
    )
    plt.plot(val_mean, label="Validation Loss")
    plt.fill_between(
        range(len(val_mean)), val_mean - val_se, val_mean + val_se, alpha=0.2
    )
    plt.title("Mean Loss Curves for Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_mean_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Aggregate and plot Predictions vs Ground Truth
try:
    all_predictions = []
    all_ground_truth = []
    for experiment_data in all_experiment_data:
        predictions = experiment_data["hyperparam_tuning_momentum"]["synthetic_data"][
            "predictions"
        ]
        ground_truth = experiment_data["hyperparam_tuning_momentum"]["synthetic_data"][
            "ground_truth"
        ]
        all_predictions.append(predictions)
        all_ground_truth.append(ground_truth)

    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)

    predictions_mean = all_predictions.mean(axis=0)
    ground_truth_mean = all_ground_truth.mean(axis=0)

    plt.figure()
    plt.scatter(ground_truth_mean, predictions_mean)
    plt.plot(
        [ground_truth_mean.min(), ground_truth_mean.max()],
        [ground_truth_mean.min(), ground_truth_mean.max()],
        "r--",
    )
    plt.title("Mean Predictions vs Ground Truth for Synthetic Data")
    plt.xlabel("Ground Truth")
    plt.ylabel("Mean Model Predictions")
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_mean_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
