import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_6e5b844a1c50472f92b4ae306e670430_proc_2524440/experiment_data.npy",
        "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_8d433054021b4ee6b2597d628dcf16c5_proc_2524441/experiment_data.npy",
        "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_e0c9598b015b4b8bb53b2a573f6dbf20_proc_2524442/experiment_data.npy",
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

# Plotting training and validation loss for different batch sizes
for batch_size in [16, 32, 64]:
    try:
        train_losses = []
        val_losses = []

        for experiment_data in all_experiment_data:
            train_losses.append(
                experiment_data["batch_size_tuning"][batch_size]["losses"]["train"]
            )
            val_losses.append(
                experiment_data["batch_size_tuning"][batch_size]["losses"]["val"]
            )

        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)

        mean_train_losses = train_losses.mean(axis=0)
        mean_val_losses = val_losses.mean(axis=0)
        sem_train_losses = train_losses.std(axis=0) / np.sqrt(len(all_experiment_data))
        sem_val_losses = val_losses.std(axis=0) / np.sqrt(len(all_experiment_data))
        epochs = np.arange(len(mean_train_losses))

        plt.figure()
        plt.plot(epochs, mean_train_losses, label="Mean Training Loss", color="blue")
        plt.fill_between(
            epochs,
            mean_train_losses - sem_train_losses,
            mean_train_losses + sem_train_losses,
            color="blue",
            alpha=0.2,
        )
        plt.plot(epochs, mean_val_losses, label="Mean Validation Loss", color="orange")
        plt.fill_between(
            epochs,
            mean_val_losses - sem_val_losses,
            mean_val_losses + sem_val_losses,
            color="orange",
            alpha=0.2,
        )
        plt.title(f"Mean Loss Curves for Batch Size {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"mean_loss_curves_batch_size_{batch_size}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {batch_size}: {e}")
        plt.close()
