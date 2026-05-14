import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_dd4e68bc5858472a97bf74c0f5992172_proc_2514467/experiment_data.npy",
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_04135784519d40cb92393f319f6706db_proc_2514470/experiment_data.npy",
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_6a1d990433ed406abac47c179e441eb6_proc_2514468/experiment_data.npy",
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

# Plotting Losses and Metrics
try:
    for act_func in experiment_data["activation_function_tuning"]["FeedbackDataset"][
        "losses"
    ]:
        train_losses = experiment_data["activation_function_tuning"]["FeedbackDataset"][
            "losses"
        ]["train"]
        val_losses = experiment_data["activation_function_tuning"]["FeedbackDataset"][
            "losses"
        ]["val"]
        epochs = range(1, len(train_losses) + 1)

        # Calculate mean and standard error
        train_loss_mean = np.mean(train_losses, axis=0)
        val_loss_mean = np.mean(val_losses, axis=0)
        train_loss_se = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        val_loss_se = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

        plt.figure()
        plt.plot(epochs, train_loss_mean, label="Mean Training Loss", color="blue")
        plt.fill_between(
            epochs,
            train_loss_mean - train_loss_se,
            train_loss_mean + train_loss_se,
            color="blue",
            alpha=0.2,
        )
        plt.plot(epochs, val_loss_mean, label="Mean Validation Loss", color="orange")
        plt.fill_between(
            epochs,
            val_loss_mean - val_loss_se,
            val_loss_mean + val_loss_se,
            color="orange",
            alpha=0.2,
        )
        plt.title(f"Aggregated Loss Curves - Activation Function: {act_func}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"FeedbackDataset_aggregated_loss_curves_{act_func}.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

try:
    train_metrics = experiment_data["activation_function_tuning"]["FeedbackDataset"][
        "metrics"
    ]["train"]
    epochs = range(1, len(train_metrics) + 1)

    # Calculate mean and standard error for metrics
    train_metrics_mean = np.mean(train_metrics, axis=0)
    train_metrics_se = np.std(train_metrics, axis=0) / np.sqrt(len(train_metrics))

    plt.figure()
    plt.plot(epochs, train_metrics_mean, label="Mean Training Metrics", color="green")
    plt.fill_between(
        epochs,
        train_metrics_mean - train_metrics_se,
        train_metrics_mean + train_metrics_se,
        color="green",
        alpha=0.2,
    )
    plt.title("Aggregated Training Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics Value")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "FeedbackDataset_aggregated_training_metrics.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metrics plot: {e}")
    plt.close()
