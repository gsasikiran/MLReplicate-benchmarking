import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_4509f906a6d242618bdca7210fd6029b_proc_2525656/experiment_data.npy",
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_399dd08fec204e3fa8ee2c19d92f6394_proc_2525657/experiment_data.npy",
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_8e94cffe610842619d21c977c6c55c14_proc_2525654/experiment_data.npy",
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
    train_losses = [
        data["synthetic_dataset"]["losses"]["train"] for data in all_experiment_data
    ]
    val_losses = [
        data["synthetic_dataset"]["losses"]["val"] for data in all_experiment_data
    ]

    # Calculate mean and standard error
    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    se_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
    se_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    plt.figure()
    plt.plot(mean_train_loss, label="Mean Train Loss")
    plt.plot(mean_val_loss, label="Mean Validation Loss")
    plt.fill_between(
        range(len(mean_train_loss)),
        mean_train_loss - se_train_loss,
        mean_train_loss + se_train_loss,
        alpha=0.2,
    )
    plt.fill_between(
        range(len(mean_val_loss)),
        mean_val_loss - se_val_loss,
        mean_val_loss + se_val_loss,
        alpha=0.2,
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curve_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    ground_truths = [
        data["synthetic_dataset"]["ground_truth"] for data in all_experiment_data
    ]
    predictions = [
        data["synthetic_dataset"]["predictions"] for data in all_experiment_data
    ]

    # Example to show how you could visualize the aggregated predictions
    mean_predictions = np.mean(predictions, axis=0)

    plt.figure()
    plt.scatter(ground_truths[0], mean_predictions)
    plt.title("Ground Truth vs Predictions (Mean)")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions (Mean)")
    plt.plot([0, 1], [0, 1], "r--")  # diagonal line for reference
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_dataset_ground_truth_vs_mean_predictions.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
