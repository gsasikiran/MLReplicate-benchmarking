import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data_path_list = [
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_3be57275e1b047e2a0a4ca54f7c89fd7_proc_2538356/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_7a839eaa768d4153a841e178274fe440_proc_2538353/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_211700fa2ba84235b42f2bcc346a88f8_proc_2538355/experiment_data.npy",
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

try:
    # Calculate mean and standard error for training losses
    losses_train = [
        exp["synthetic_dataset"]["losses"]["train"] for exp in all_experiment_data
    ]
    mean_loss = np.mean(losses_train, axis=0)
    std_error_loss = np.std(losses_train, axis=0) / np.sqrt(len(losses_train))

    # Plotting mean training loss
    plt.figure()
    plt.errorbar(
        range(len(mean_loss)),
        mean_loss,
        yerr=std_error_loss,
        label="Mean Training Loss",
        capsize=5,
    )
    plt.title("Mean Training Loss over Epochs for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Mean_Training_Loss_Synthetic_Dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")
    plt.close()

try:
    # Calculate mean and standard error for training accuracy
    accuracy_train = [
        exp["synthetic_dataset"]["metrics"]["train"] for exp in all_experiment_data
    ]
    mean_accuracy = np.mean(accuracy_train, axis=0)
    std_error_accuracy = np.std(accuracy_train, axis=0) / np.sqrt(len(accuracy_train))

    # Plotting mean training accuracy
    plt.figure()
    plt.errorbar(
        range(len(mean_accuracy)),
        mean_accuracy,
        yerr=std_error_accuracy,
        label="Mean Training Accuracy",
        color="orange",
        capsize=5,
    )
    plt.title("Mean Training Accuracy over Epochs for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "Mean_Training_Accuracy_Synthetic_Dataset.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean training accuracy plot: {e}")
    plt.close()
