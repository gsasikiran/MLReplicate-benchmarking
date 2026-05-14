import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_3e505d60122c4991b42d438d4b30eab5_proc_2532938/experiment_data.npy",
    "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_9f53092111204d0189f13a11e92a06a9_proc_2532939/experiment_data.npy",
    "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_b2cc2265257047699860fa71574a962e_proc_2532941/experiment_data.npy",
]

try:
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate and plot training and validation losses
try:
    plt.figure()
    train_losses = []
    val_losses = []

    for experiment in all_experiment_data:
        train_losses.append(
            experiment["hyperparam_tuning_num_epochs"]["sudoku"]["losses"]["train"]
        )
        val_losses.append(
            experiment["hyperparam_tuning_num_epochs"]["sudoku"]["losses"]["val"]
        )

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    epochs = range(1, train_losses.shape[1] + 1)

    train_means = train_losses.mean(axis=0)
    train_sems = train_losses.std(axis=0) / np.sqrt(train_losses.shape[0])
    val_means = val_losses.mean(axis=0)
    val_sems = val_losses.std(axis=0) / np.sqrt(val_losses.shape[0])

    plt.errorbar(
        epochs, train_means, yerr=train_sems, label="Training Loss", capsize=3, fmt="-o"
    )
    plt.errorbar(
        epochs, val_means, yerr=val_sems, label="Validation Loss", capsize=3, fmt="-o"
    )

    plt.title("Mean Losses with SEM Over Epochs - Sudoku Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_mean_losses_sem.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()
