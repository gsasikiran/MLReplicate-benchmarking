import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_6d70540135e6416faf20b64ad2f0b888_proc_2519725/experiment_data.npy",
        "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_9de36bc27f9143fdba97c207e8b14d7d_proc_2519723/experiment_data.npy",
        "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_ff735e45e6bc4d57a6f1ae97d2737f0c_proc_2519724/experiment_data.npy",
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
    metrics_train = np.array(
        [
            exp["hyperparam_tuning_batch_size"]["RQS"]["metrics"]["train"]
            for exp in all_experiment_data
        ]
    )
    metrics_val = np.array(
        [
            exp["hyperparam_tuning_batch_size"]["RQS"]["metrics"]["val"]
            for exp in all_experiment_data
        ]
    )
    epochs = np.arange(1, metrics_train.shape[1] + 1)

    mean_train = np.mean(metrics_train, axis=0)
    mean_val = np.mean(metrics_val, axis=0)
    stderr_train = np.std(metrics_train, axis=0) / np.sqrt(metrics_train.shape[0])
    stderr_val = np.std(metrics_val, axis=0) / np.sqrt(metrics_val.shape[0])

    plt.figure()
    plt.plot(epochs, mean_train, label="Mean Training Metric")
    plt.plot(epochs, mean_val, label="Mean Validation Metric")
    plt.fill_between(
        epochs, mean_train - stderr_train, mean_train + stderr_train, alpha=0.2
    )
    plt.fill_between(epochs, mean_val - stderr_val, mean_val + stderr_val, alpha=0.2)
    plt.title("Mean Training and Validation Metrics with Error Bars")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "RQS_mean_training_validation_metrics_with_error_bars.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

try:
    losses_train = np.array(
        [
            exp["hyperparam_tuning_batch_size"]["RQS"]["losses"]["train"]
            for exp in all_experiment_data
        ]
    )
    losses_val = np.array(
        [
            exp["hyperparam_tuning_batch_size"]["RQS"]["losses"]["val"]
            for exp in all_experiment_data
        ]
    )

    mean_loss_train = np.mean(losses_train, axis=0)
    mean_loss_val = np.mean(losses_val, axis=0)
    stderr_loss_train = np.std(losses_train, axis=0) / np.sqrt(losses_train.shape[0])
    stderr_loss_val = np.std(losses_val, axis=0) / np.sqrt(losses_val.shape[0])

    plt.figure()
    plt.plot(epochs, mean_loss_train, label="Mean Training Loss")
    plt.plot(epochs, mean_loss_val, label="Mean Validation Loss")
    plt.fill_between(
        epochs,
        mean_loss_train - stderr_loss_train,
        mean_loss_train + stderr_loss_train,
        alpha=0.2,
    )
    plt.fill_between(
        epochs,
        mean_loss_val - stderr_loss_val,
        mean_loss_val + stderr_loss_val,
        alpha=0.2,
    )
    plt.title("Mean Training and Validation Loss with Error Bars")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "RQS_mean_training_validation_losses_with_error_bars.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating losses plot: {e}")
    plt.close()
