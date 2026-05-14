import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_09978816ee404c0db28f5bcd1410b861_proc_2515973/experiment_data.npy",
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_ab56c5e9f0e641dbab6fc724bb482f09_proc_2515972/experiment_data.npy",
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_23a6efcb79064bec9fee99e9e28fdcf2_proc_2515971/experiment_data.npy",
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

# Plot aggregated training loss
try:
    losses = [data["synthetic_data"]["losses"]["train"] for data in all_experiment_data]
    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)
    se_losses = std_losses / np.sqrt(len(all_experiment_data))

    epochs = np.arange(len(mean_losses))
    plt.figure()
    plt.plot(epochs, mean_losses, label="Mean Training Loss", color="blue")
    plt.fill_between(
        epochs,
        mean_losses - se_losses,
        mean_losses + se_losses,
        color="blue",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Aggregated Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_aggregated_training_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training loss plot: {e}")
    plt.close()

# Plot aggregated training RQS
try:
    rqs = [data["synthetic_data"]["metrics"]["train"] for data in all_experiment_data]
    mean_rqs = np.mean(rqs, axis=0)
    std_rqs = np.std(rqs, axis=0)
    se_rqs = std_rqs / np.sqrt(len(all_experiment_data))

    plt.figure()
    plt.plot(epochs, mean_rqs, label="Mean Training RQS", color="orange")
    plt.fill_between(
        epochs,
        mean_rqs - se_rqs,
        mean_rqs + se_rqs,
        color="orange",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Aggregated Training RQS Curve")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_aggregated_training_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training RQS plot: {e}")
    plt.close()
