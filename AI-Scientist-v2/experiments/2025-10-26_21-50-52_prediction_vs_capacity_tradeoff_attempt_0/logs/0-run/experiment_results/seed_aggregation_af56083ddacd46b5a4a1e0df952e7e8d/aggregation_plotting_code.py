markdown
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_paths = [
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_f8d621b26253490cb95cab4fe055eb05_proc_2538789/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_7231ce06afdd4b17b10d684df7513beb_proc_2538788/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_1f43b348c4cd4423a3c5b095b8f01a06_proc_2538791/experiment_data.npy",
    ]

    all_losses = []
    all_accuracies = []

    for experiment_data_path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_losses.append(
            experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
                "losses"
            ]["train"]
        )
        all_accuracies.append(
            experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
                "metrics"
            ]["train"]
        )

    all_losses = np.array(all_losses)
    all_accuracies = np.array(all_accuracies)

    mean_losses = np.mean(all_losses, axis=0)
    se_losses = np.std(all_losses, axis=0) / np.sqrt(all_losses.shape[0])

    mean_accuracies = np.mean(all_accuracies, axis=0)
    se_accuracies = np.std(all_accuracies, axis=0) / np.sqrt(all_accuracies.shape[0])

    try:
        # Plotting Mean Training Loss with Error Bars
        plt.figure()
        plt.errorbar(
            range(len(mean_losses)),
            mean_losses,
            yerr=se_losses,
            label="Mean Loss ± SE",
            capsize=5,
        )
        plt.title("Mean Training Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_dataset_mean_training_loss.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating mean training loss plot: {e}")
        plt.close()

    try:
        # Plotting Mean Training Accuracy with Error Bars
        plt.figure()
        plt.errorbar(
            range(len(mean_accuracies)),
            mean_accuracies,
            yerr=se_accuracies,
            label="Mean Accuracy ± SE",
            capsize=5,
        )
        plt.title("Mean Training Accuracy over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_dataset_mean_training_accuracy.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating mean training accuracy plot: {e}")
        plt.close()

except Exception as e:
    print(f"Error loading experiment data: {e}")
