import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_fbdbeee9bf704a30a798c2e7ad9c3071_proc_2517977/experiment_data.npy",
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_f4eefac8215e42e59784d04bce042439_proc_2517976/experiment_data.npy",
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_35600d1e7d9b4a4caa51dea44b1469ca_proc_2517978/experiment_data.npy",
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


def plot_with_error_bars(data, title, ylabel, filename):
    means = np.mean(data, axis=0)
    std_errors = np.std(data, axis=0) / np.sqrt(data.shape[0])

    try:
        plt.figure()
        plt.plot(means, label="Mean", color="blue")
        plt.fill_between(
            range(len(means)),
            means - std_errors,
            means + std_errors,
            color="blue",
            alpha=0.2,
            label="Standard Error",
        )
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(os.path.join(working_dir, filename))
        plt.close()
    except Exception as e:
        print(f"Error creating plot '{title}': {e}")
        plt.close()


# Plot training loss
plot_with_error_bars(
    np.array([data["peer_review"]["losses"]["train"] for data in all_experiment_data]),
    "Peer Review Experiment - Training Loss",
    "Loss",
    "peer_review_training_loss.png",
)

# Plot validation loss
plot_with_error_bars(
    np.array([data["peer_review"]["losses"]["val"] for data in all_experiment_data]),
    "Peer Review Experiment - Validation Loss",
    "Loss",
    "peer_review_validation_loss.png",
)

# Plot RQI metrics
plot_with_error_bars(
    np.array([data["peer_review"]["metrics"]["train"] for data in all_experiment_data]),
    "Peer Review Experiment - RQI Metrics",
    "RQI",
    "peer_review_rqi_metrics.png",
)
