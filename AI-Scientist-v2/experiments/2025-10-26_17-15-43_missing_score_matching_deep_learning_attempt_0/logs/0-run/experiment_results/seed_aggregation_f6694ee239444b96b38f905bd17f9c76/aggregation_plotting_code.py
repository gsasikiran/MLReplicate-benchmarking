import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_paths = [
    "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_dc4a5129799841e9bbe5bae57f85acd5_proc_2535213/experiment_data.npy",
    "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_a698838c0e8c4966b02ff15b58b2854e_proc_2535212/experiment_data.npy",
    "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_d9c1d8dc1767494fa831f1554bcd2678_proc_2535214/experiment_data.npy",
]

all_experiment_data = []

try:
    for experiment_data_path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregating and plotting the results
for method in all_experiment_data[0]["imputation_methods"]:
    try:
        train_losses = []
        val_losses = []
        mdie_values = []

        for data in all_experiment_data:
            train_losses.append(data["imputation_methods"][method]["losses"]["train"])
            val_losses.append(data["imputation_methods"][method]["losses"]["val"])
            mdie_values.append(data["imputation_methods"][method]["mdie"])

        # Mean and SEM for training loss
        train_mean = np.mean(train_losses, axis=0)
        train_sem = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))

        # Mean and SEM for validation loss
        val_mean = np.mean(val_losses, axis=0)
        val_sem = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

        # Mean and SEM for MDIE
        mdie_mean = np.mean(mdie_values, axis=0)
        mdie_sem = np.std(mdie_values, axis=0) / np.sqrt(len(mdie_values))

        # Plot training loss
        plt.figure()
        plt.plot(train_mean, label="Mean Training Loss")
        plt.fill_between(
            range(len(train_mean)),
            train_mean - train_sem,
            train_mean + train_sem,
            alpha=0.2,
            label="SEM",
        )
        plt.plot(val_mean, label="Mean Validation Loss")
        plt.fill_between(
            range(len(val_mean)), val_mean - val_sem, val_mean + val_sem, alpha=0.2
        )
        plt.title(f"Loss Curves for {method.capitalize()} Imputation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{method}.png"))
        plt.close()

    except Exception as e:
        print(f"Error creating plot for {method}: {e}")
        plt.close()

    try:
        # Plot MDIE
        plt.figure()
        plt.plot(mdie_mean, label="Mean MDIE", color="orange")
        plt.fill_between(
            range(len(mdie_mean)),
            mdie_mean - mdie_sem,
            mdie_mean + mdie_sem,
            alpha=0.2,
            label="SEM",
        )
        plt.title(f"MDIE for {method.capitalize()} Imputation")
        plt.xlabel("Epochs")
        plt.ylabel("MDIE")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"mdie_{method}.png"))
        plt.close()

    except Exception as e:
        print(f"Error creating MDIE plot for {method}: {e}")
        plt.close()
