import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_eb767ca6376945abb5f69ffbb141611b_proc_2513375/experiment_data.npy",
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_6cd86514e63548d2a555a82bb24a5702_proc_2513376/experiment_data.npy",
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_1e6a7f1febd14b5ca572dee89eef633b_proc_2513375/experiment_data.npy",
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

for act_name in all_experiment_data[0]["activation_function_tuning"].keys():
    try:
        train_losses = []
        val_losses = []

        for experiment_data in all_experiment_data:
            train_losses.append(
                experiment_data["activation_function_tuning"][act_name]["losses"][
                    "train"
                ]
            )
            val_losses.append(
                experiment_data["activation_function_tuning"][act_name]["losses"]["val"]
            )

        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)

        mean_train_loss = np.mean(train_losses, axis=0)
        mean_val_loss = np.mean(val_losses, axis=0)
        sem_train_loss = np.std(train_losses, axis=0) / np.sqrt(train_losses.shape[0])
        sem_val_loss = np.std(val_losses, axis=0) / np.sqrt(val_losses.shape[0])

        plt.figure()
        plt.plot(mean_train_loss, label="Mean Train Loss")
        plt.fill_between(
            range(len(mean_train_loss)),
            mean_train_loss - sem_train_loss,
            mean_train_loss + sem_train_loss,
            alpha=0.2,
        )
        plt.plot(mean_val_loss, label="Mean Validation Loss")
        plt.fill_between(
            range(len(mean_val_loss)),
            mean_val_loss - sem_val_loss,
            mean_val_loss + sem_val_loss,
            alpha=0.2,
        )
        plt.title(f"{act_name} Activation Function Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_mean_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name}: {e}")
        plt.close()
