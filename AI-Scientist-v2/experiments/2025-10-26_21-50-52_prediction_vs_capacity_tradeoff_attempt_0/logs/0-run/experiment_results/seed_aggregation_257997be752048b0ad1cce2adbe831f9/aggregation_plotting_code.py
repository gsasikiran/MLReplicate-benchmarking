import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_06f47005e7ff4abda0cd648e11155724_proc_2538464/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_1fb5c95a3b73440c8c165cd930f4e52f_proc_2538462/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_4a2841a4476c49a88854022224b38799_proc_2538461/experiment_data.npy",
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
    # Aggregating Training Loss
    epochs = np.arange(
        len(
            experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
                "losses"
            ]["train"]
        )
    )
    intervals = [epoch for epoch in epochs if epoch % 10 == 0]  # Choose every 10 epochs
    losses = experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
        "losses"
    ]["train"]
    mean_losses = np.mean(losses[intervals])
    std_error_losses = np.std(losses[intervals]) / np.sqrt(len(intervals))

    plt.figure()
    plt.errorbar(
        intervals, losses[intervals], yerr=std_error_losses, label="Mean ± SE", fmt="-o"
    )
    plt.title("Mean Training Loss over Selected Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss_mean.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Aggregating Training Accuracy
    metrics = experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    mean_accuracy = np.mean(metrics[intervals])
    std_error_accuracy = np.std(metrics[intervals]) / np.sqrt(len(intervals))

    plt.figure()
    plt.errorbar(
        intervals,
        metrics[intervals],
        yerr=std_error_accuracy,
        label="Mean ± SE",
        fmt="-o",
    )
    plt.title("Mean Training Accuracy over Selected Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_accuracy_mean.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
