import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_460e9ee14ac14417819bba390187c096_proc_2520251/experiment_data.npy",
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_3edc659327384eb6b7bd558bb189afbe_proc_2520250/experiment_data.npy",
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_c62d2c7a06ba486394439289a2fbe518_proc_2520248/experiment_data.npy",
]

all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")


# Function to aggregate and plot
def plot_with_error_bars(data_train, data_val, title, ylabel, plot_name):
    try:
        epochs = np.arange(len(data_train))
        mean_train = np.mean(data_train, axis=0)
        mean_val = np.mean(data_val, axis=0)
        se_train = np.std(data_train, axis=0) / np.sqrt(data_train.shape[0])
        se_val = np.std(data_val, axis=0) / np.sqrt(data_val.shape[0])

        plt.figure()
        plt.plot(epochs, mean_train, label="Mean Train", color="blue")
        plt.fill_between(
            epochs,
            mean_train - se_train,
            mean_train + se_train,
            color="blue",
            alpha=0.2,
        )
        plt.plot(epochs, mean_val, label="Mean Validation", color="orange")
        plt.fill_between(
            epochs, mean_val - se_val, mean_val + se_val, color="orange", alpha=0.2
        )
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(os.path.join(working_dir, plot_name))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {title}: {e}")
        plt.close()


# Aggregated Training and Validation Losses
losses_train = np.array(
    [
        data["hyperparam_tuning_lr"]["RQS"]["losses"]["train"]
        for data in all_experiment_data
    ]
)
losses_val = np.array(
    [
        data["hyperparam_tuning_lr"]["RQS"]["losses"]["val"]
        for data in all_experiment_data
    ]
)
plot_with_error_bars(
    losses_train,
    losses_val,
    "Aggregated Training and Validation Losses",
    "Loss",
    "Experiment_RQS_Aggregated_Training_Validation_Losses.png",
)

# Aggregated Training and Validation Metrics
metrics_train = np.array(
    [
        data["hyperparam_tuning_lr"]["RQS"]["metrics"]["train"]
        for data in all_experiment_data
    ]
)
metrics_val = np.array(
    [
        data["hyperparam_tuning_lr"]["RQS"]["metrics"]["val"]
        for data in all_experiment_data
    ]
)
plot_with_error_bars(
    metrics_train,
    metrics_val,
    "Aggregated Training and Validation Metrics",
    "Metric Value",
    "Experiment_RQS_Aggregated_Training_Validation_Metrics.png",
)
