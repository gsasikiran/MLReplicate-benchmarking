import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_19-10-36_creative_limits_lms_attempt_0/logs/0-run/experiment_results/experiment_7367eceed4f84d78b53a62973e19fdcb_proc_2537106/experiment_data.npy",
        "experiments/2025-10-26_19-10-36_creative_limits_lms_attempt_0/logs/0-run/experiment_results/experiment_d3edfae866be48f88021a47d1b82adaf_proc_2537108/experiment_data.npy",
        "experiments/2025-10-26_19-10-36_creative_limits_lms_attempt_0/logs/0-run/experiment_results/experiment_aec5f4190076474eae85caf6ed80af6c_proc_2537107/experiment_data.npy",
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

# Extract training losses and metrics
training_losses = []
training_metrics = []

for exp_data in all_experiment_data:
    training_losses.append(
        exp_data["input_noise_variation"]["synthetic_dataset"]["losses"]["train"]
    )
    training_metrics.append(
        exp_data["input_noise_variation"]["synthetic_dataset"]["metrics"]["train"]
    )

# Calculate mean and standard error for training losses
mean_losses = np.mean(training_losses, axis=0)
std_errors_losses = np.std(training_losses, axis=0) / np.sqrt(len(all_experiment_data))

# Calculate mean and standard error for training CODS
mean_metrics = np.mean(training_metrics, axis=0)
std_errors_metrics = np.std(training_metrics, axis=0) / np.sqrt(
    len(all_experiment_data)
)

# Epochs
epochs = np.arange(1, len(mean_losses) + 1)

# Plot Mean Training Loss
try:
    plt.figure()
    plt.plot(epochs, mean_losses, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_losses - std_errors_losses,
        mean_losses + std_errors_losses,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Training Loss Over Epochs (Synthetic Dataset)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")
    plt.close()

# Plot Mean Training CODS
try:
    plt.figure()
    plt.plot(epochs, mean_metrics, label="Mean CODS", color="orange")
    plt.fill_between(
        epochs,
        mean_metrics - std_errors_metrics,
        mean_metrics + std_errors_metrics,
        alpha=0.2,
        color="orange",
        label="Standard Error",
    )
    plt.title("Mean Training CODS Over Epochs (Synthetic Dataset)")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_training_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training CODS plot: {e}")
    plt.close()
