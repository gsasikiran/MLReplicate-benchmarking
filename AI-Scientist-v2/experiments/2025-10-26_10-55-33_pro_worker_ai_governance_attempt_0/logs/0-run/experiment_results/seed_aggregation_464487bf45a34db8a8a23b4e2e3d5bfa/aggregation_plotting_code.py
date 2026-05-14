import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

# List of experiment paths
experiment_data_path_list = [
    "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_ca5793d4c81f44c0b7291f30b2252f2c_proc_2523981/experiment_data.npy",
    "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_b6a847d54a2347e38bc9c935e05febd6_proc_2523984/experiment_data.npy",
    "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_ca4dc208f4b448ca935615ffe0f8641d_proc_2523983/experiment_data.npy",
]

all_experiment_data = []

# Load experiment data
try:
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Prepare to store mean and standard error for losses
batch_sizes = [16, 32, 64]

for batch_size in batch_sizes:
    try:
        train_losses_list = []
        val_losses_list = []

        for exp_data in all_experiment_data:
            train_losses = exp_data["batch_size_tuning"][batch_size]["losses"]["train"]
            val_losses = exp_data["batch_size_tuning"][batch_size]["losses"]["val"]
            train_losses_list.append(train_losses)
            val_losses_list.append(val_losses)

        # Calculate mean and standard error
        train_losses_mean = np.mean(train_losses_list, axis=0)
        val_losses_mean = np.mean(val_losses_list, axis=0)
        train_losses_se = np.std(train_losses_list, axis=0) / np.sqrt(
            len(train_losses_list)
        )
        val_losses_se = np.std(val_losses_list, axis=0) / np.sqrt(len(val_losses_list))
        epochs = range(len(train_losses_mean))

        # Plot mean with error bars
        plt.figure()
        plt.plot(epochs, train_losses_mean, label="Training Loss")
        plt.fill_between(
            epochs,
            train_losses_mean - train_losses_se,
            train_losses_mean + train_losses_se,
            color="blue",
            alpha=0.1,
        )
        plt.plot(epochs, val_losses_mean, label="Validation Loss")
        plt.fill_between(
            epochs,
            val_losses_mean - val_losses_se,
            val_losses_mean + val_losses_se,
            color="orange",
            alpha=0.1,
        )
        plt.title(f"Average Loss Curves for Batch Size {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"avg_loss_curves_batch_size_{batch_size}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {batch_size}: {e}")
        plt.close()
