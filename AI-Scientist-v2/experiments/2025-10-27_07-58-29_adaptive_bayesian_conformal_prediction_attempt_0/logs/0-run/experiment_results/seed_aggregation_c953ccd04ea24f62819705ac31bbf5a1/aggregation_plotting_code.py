import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_f0c00d0850464601988fd03deb8d7355_proc_2543342/experiment_data.npy",
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_e02a7c531c224cf89117d6e1e9e98a2a_proc_2543341/experiment_data.npy",
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_705f829ba2f944059c465348152d3378_proc_2543339/experiment_data.npy",
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

# Aggregate training vs validation loss
try:
    train_losses = np.array(
        [exp["synthetic_data"]["losses"]["train"] for exp in all_experiment_data]
    )
    val_losses = np.array(
        [exp["synthetic_data"]["losses"]["val"] for exp in all_experiment_data]
    )

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_val_losses = np.mean(val_losses, axis=0)

    std_train_losses = np.std(train_losses, axis=0)
    std_val_losses = np.std(val_losses, axis=0)

    plt.figure()
    plt.plot(mean_train_losses, label="Mean Training Loss")
    plt.plot(mean_val_losses, label="Mean Validation Loss")
    plt.fill_between(
        range(len(mean_train_losses)),
        mean_train_losses - std_train_losses,
        mean_train_losses + std_train_losses,
        alpha=0.2,
    )
    plt.fill_between(
        range(len(mean_val_losses)),
        mean_val_losses - std_val_losses,
        mean_val_losses + std_val_losses,
        alpha=0.2,
    )
    plt.title("Aggregated Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve plot: {e}")
    plt.close()

# Aggregate predictions vs ground truth for validation
try:
    ground_truths = np.array(
        [exp["synthetic_data"]["ground_truth"] for exp in all_experiment_data]
    )
    predictions = np.array(
        [exp["synthetic_data"]["predictions"] for exp in all_experiment_data]
    )

    mean_ground_truth = np.mean(ground_truths, axis=0)
    mean_predictions = np.mean(predictions, axis=0)

    plt.figure()
    plt.scatter(
        mean_ground_truth, mean_predictions[0], label="Mean Predictions", alpha=0.5
    )
    plt.plot(mean_ground_truth, mean_ground_truth, "r-", label="Ideal Prediction Line")
    plt.title("Aggregated Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Mean Predicted Values")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated predictions plot: {e}")
    plt.close()
