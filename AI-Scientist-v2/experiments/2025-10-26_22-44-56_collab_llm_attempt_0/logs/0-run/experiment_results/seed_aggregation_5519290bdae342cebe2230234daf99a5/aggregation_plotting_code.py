import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data_path_list = [
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_ac13d0d642d44d178edcfeb474185bfa_proc_2539573/experiment_data.npy",
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_64b545875d9545ffb24e2395b26cf63c_proc_2539574/experiment_data.npy",
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
    # Aggregate losses and metrics
    train_losses = [
        data["multi_turn_interactions"]["losses"]["train"]
        for data in all_experiment_data
    ]
    train_metrics = [
        data["multi_turn_interactions"]["metrics"]["train"]
        for data in all_experiment_data
    ]

    # Calculate mean and standard error
    mean_losses = np.mean(train_losses, axis=0)
    std_error_losses = np.std(train_losses, axis=0) / np.sqrt(len(all_experiment_data))

    mean_metrics = np.mean(train_metrics, axis=0)
    std_error_metrics = np.std(train_metrics, axis=0) / np.sqrt(
        len(all_experiment_data)
    )

    epochs = np.arange(1, len(mean_losses) + 1)

    # Plot aggregated training losses
    plt.figure()
    plt.plot(epochs, mean_losses, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_losses - std_error_losses,
        mean_losses + std_error_losses,
        color="b",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Aggregated Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "aggregated_multi_turn_interactions_training_loss.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training loss plot: {e}")

try:
    # Plot aggregated training metrics
    plt.figure()
    plt.plot(epochs, mean_metrics, label="Mean Training UES")
    plt.fill_between(
        epochs,
        mean_metrics - std_error_metrics,
        mean_metrics + std_error_metrics,
        color="r",
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Aggregated Training UES Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("UES")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "aggregated_multi_turn_interactions_training_ues.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training UES plot: {e}")
