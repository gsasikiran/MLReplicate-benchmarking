import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_01524950d38441d38ce751ad3d39baa5_proc_2539011/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_79bd369830d24c2e87ccb1aca7d04e20_proc_2539014/experiment_data.npy",
        "experiments/2025-10-26_21-50-52_prediction_vs_capacity_tradeoff_attempt_0/logs/0-run/experiment_results/experiment_30d608b0d14e4ddd95753d7d7b6e88b5_proc_2539012/experiment_data.npy",
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

# Plotting Training Losses
try:
    plt.figure()
    train_losses = [
        exp["ablation_model_architecture"]["synthetic_dataset"]["losses"]["train"]
        for exp in all_experiment_data
    ]
    mean_losses = np.mean(train_losses, axis=0)
    std_losses = np.std(train_losses, axis=0)
    epochs = range(len(mean_losses))
    plt.plot(epochs, mean_losses, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_losses - std_losses,
        mean_losses + std_losses,
        alpha=0.2,
        label="Std Error",
    )
    plt.title("Training Losses Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_loss_aggregated.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plotting Training Accuracies
try:
    plt.figure()
    train_accuracies = [
        exp["ablation_model_architecture"]["synthetic_dataset"]["metrics"]["train"]
        for exp in all_experiment_data
    ]
    mean_accuracies = np.mean(train_accuracies, axis=0)
    std_accuracies = np.std(train_accuracies, axis=0)
    epochs = range(len(mean_accuracies))
    plt.plot(epochs, mean_accuracies, label="Mean Training Accuracy")
    plt.fill_between(
        epochs,
        mean_accuracies - std_accuracies,
        mean_accuracies + std_accuracies,
        alpha=0.2,
        label="Std Error",
    )
    plt.title("Training Accuracies Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_accuracy_aggregated.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
