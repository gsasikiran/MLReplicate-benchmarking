import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_95cebbaa85f046949df782462413e260_proc_2532777/experiment_data.npy",
        "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_3788f2d2257e412bb38da226d18ed835_proc_2532776/experiment_data.npy",
        "experiments/2025-10-26_16-18-51_adaptive_inference_for_masked_diffusion_attempt_0/logs/0-run/experiment_results/experiment_b84ce7167a5d4b09b16d40304d1fefcb_proc_2532775/experiment_data.npy",
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
    # Calculate means and standard errors for losses
    train_losses = [data["sudoku"]["losses"]["train"] for data in all_experiment_data]
    val_losses = [data["sudoku"]["losses"]["val"] for data in all_experiment_data]

    mean_train_losses = np.mean(train_losses, axis=0)
    se_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))

    mean_val_losses = np.mean(val_losses, axis=0)
    se_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    # Plot training loss with error bars
    plt.figure()
    plt.errorbar(
        range(len(mean_train_losses)),
        mean_train_losses,
        yerr=se_train_losses,
        label="Training Loss",
        fmt="-o",
    )
    plt.title("Sudoku Dataset - Mean Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_mean_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")
    plt.close()

try:
    # Plot validation loss with error bars
    plt.figure()
    plt.errorbar(
        range(len(mean_val_losses)),
        mean_val_losses,
        yerr=se_val_losses,
        label="Validation Loss",
        color="orange",
        fmt="-o",
    )
    plt.title("Sudoku Dataset - Mean Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_mean_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean validation loss plot: {e}")
    plt.close()
