import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data_path_list = [
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_88fcb4684e774585872e67927fdd821c_proc_2513756/experiment_data.npy",
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_3faf594cf81644498d17863b187a000f_proc_2513755/experiment_data.npy",
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_885ab4de36fc4418bae018055fa3f2e7_proc_2513756/experiment_data.npy",
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

# Plot training and validation loss curves
for i in range(1, 4):
    try:
        dataset_name = f"dataset_{i}"
        train_losses = [
            data[dataset_name]["losses"]["train"] for data in all_experiment_data
        ]
        val_losses = [
            data[dataset_name]["losses"]["val"] for data in all_experiment_data
        ]

        # Calculate mean and standard error
        train_mean = np.mean(train_losses, axis=0)
        val_mean = np.mean(val_losses, axis=0)
        train_se = np.std(train_losses, axis=0) / np.sqrt(len(all_experiment_data))
        val_se = np.std(val_losses, axis=0) / np.sqrt(len(all_experiment_data))

        plt.figure()
        epochs = np.arange(len(train_mean))
        plt.plot(epochs, train_mean, label="Training Loss", color="blue")
        plt.plot(epochs, val_mean, label="Validation Loss", color="orange")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            color="blue",
            alpha=0.2,
        )
        plt.fill_between(
            epochs, val_mean - val_se, val_mean + val_se, color="orange", alpha=0.2
        )
        plt.title(f"{dataset_name} Loss Curves with Mean and SE")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_name}_loss_curves_mean_se.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {e}")
        plt.close()
