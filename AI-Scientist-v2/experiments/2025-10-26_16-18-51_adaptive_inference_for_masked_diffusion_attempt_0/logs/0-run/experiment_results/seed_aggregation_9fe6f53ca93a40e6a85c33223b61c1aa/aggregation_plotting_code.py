import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data_path_list = [
        "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_b9810a2bc1644e9990a5a99219b190ea_proc_2533253/experiment_data.npy",
        "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_6319e3209db2475dbf4f782ff4ca5154_proc_2533255/experiment_data.npy",
        "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_ed9065f06e1246dfb46e5b79a26887fb_proc_2533254/experiment_data.npy",
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
    plt.figure()
    train_losses = [
        exp["hyperparam_tuning_num_epochs"]["sudoku"]["losses"]["train"]
        for exp in all_experiment_data
    ]
    val_losses = [
        exp["hyperparam_tuning_num_epochs"]["sudoku"]["losses"]["val"]
        for exp in all_experiment_data
    ]
    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    se_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
    se_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    epochs = range(1, len(mean_train_loss) + 1)
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

    plt.title("Mean Losses Over Epochs - Sudoku Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_mean_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()
