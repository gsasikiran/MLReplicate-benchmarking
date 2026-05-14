import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_1012b55afa6d4ed3b0b18332a09cc075_proc_2514348/experiment_data.npy",
    "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_117cb42d92204943af3e4ab345fcf9b5_proc_2514347/experiment_data.npy",
    "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_cf9466eaaca347d49c7e7631807e6dbe_proc_2514349/experiment_data.npy",
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

# Extracting and plotting the data
losses_train, losses_val, metrics_train = [], [], []
for exp in all_experiment_data:
    losses_train.append(exp["peer_review_feedback"]["losses"]["train"])
    losses_val.append(exp["peer_review_feedback"]["losses"]["val"])
    metrics_train.append(exp["peer_review_feedback"]["metrics"]["train"])

losses_train_mean = np.mean(losses_train, axis=0)
losses_val_mean = np.mean(losses_val, axis=0)
metrics_train_mean = np.mean(metrics_train, axis=0)

losses_train_sem = np.std(losses_train, axis=0) / np.sqrt(len(all_experiment_data))
losses_val_sem = np.std(losses_val, axis=0) / np.sqrt(len(all_experiment_data))
metrics_train_sem = np.std(metrics_train, axis=0) / np.sqrt(len(all_experiment_data))

try:
    plt.figure()
    plt.plot(losses_train_mean, label="Mean Training Loss")
    plt.fill_between(
        range(len(losses_train_mean)),
        losses_train_mean - losses_train_sem,
        losses_train_mean + losses_train_sem,
        alpha=0.3,
    )
    plt.plot(losses_val_mean, label="Mean Validation Loss")
    plt.fill_between(
        range(len(losses_val_mean)),
        losses_val_mean - losses_val_sem,
        losses_val_mean + losses_val_sem,
        alpha=0.3,
    )
    plt.title("Mean Training and Validation Losses for Peer Review Feedback")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_feedback_mean_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(metrics_train_mean, label="Mean Training Metrics")
    plt.fill_between(
        range(len(metrics_train_mean)),
        metrics_train_mean - metrics_train_sem,
        metrics_train_mean + metrics_train_sem,
        alpha=0.3,
    )
    plt.title("Mean Training Metrics for Peer Review Feedback")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_feedback_mean_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
