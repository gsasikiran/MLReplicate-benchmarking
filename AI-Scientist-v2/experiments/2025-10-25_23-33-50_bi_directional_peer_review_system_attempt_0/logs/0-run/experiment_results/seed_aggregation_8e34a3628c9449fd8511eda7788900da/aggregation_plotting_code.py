import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_788724238c2541ebb7145f704f922c99_proc_2518848/experiment_data.npy",
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_f0e2cc4cd4d8434a834a0a0815aa293f_proc_2518847/experiment_data.npy",
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_874ff9425c2e4af0b6f2bc64b862b376_proc_2518845/experiment_data.npy",
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

for noise_level in ["low_noise", "medium_noise", "high_noise"]:
    try:
        train_losses = []
        val_losses = []
        for exp_data in all_experiment_data:
            train_losses.append(
                exp_data["multi_synthetic"][noise_level]["losses"]["train"]
            )
            val_losses.append(exp_data["multi_synthetic"][noise_level]["losses"]["val"])

        mean_train_loss = np.mean(train_losses, axis=0)
        mean_val_loss = np.mean(val_losses, axis=0)
        se_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        se_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))
        epochs = list(range(1, len(mean_train_loss) + 1))

        plt.figure()
        plt.plot(epochs, mean_train_loss, label="Mean Training Loss")
        plt.plot(epochs, mean_val_loss, label="Mean Validation Loss")
        plt.fill_between(
            epochs,
            mean_train_loss - se_train_loss,
            mean_train_loss + se_train_loss,
            alpha=0.1,
        )
        plt.fill_between(
            epochs, mean_val_loss - se_val_loss, mean_val_loss + se_val_loss, alpha=0.1
        )
        plt.title(f"{noise_level} Dataset Loss Curves (Mean with SE)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{noise_level}_aggregated_loss_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {noise_level}: {e}")
        plt.close()
