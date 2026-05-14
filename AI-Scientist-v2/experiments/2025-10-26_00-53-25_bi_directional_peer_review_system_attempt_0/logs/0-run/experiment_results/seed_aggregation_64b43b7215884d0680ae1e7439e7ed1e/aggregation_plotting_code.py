import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_427bf9185f1a4d2daefe724a7381e4ad_proc_2520697/experiment_data.npy",
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_9b4090b1d2b944f1b7ffe303074709a6_proc_2520695/experiment_data.npy",
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_9a65a5245f4445b0970b556955e887e1_proc_2520694/experiment_data.npy",
]
all_experiment_data = []

for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data: {e}")

# Aggregate results
noise_levels = list(all_experiment_data[0]["noise_robustness"].keys())
train_metrics = []
val_metrics = []
train_losses = []
val_losses = []

for noise in noise_levels:
    # Collect values from each experiment
    for exp_data in all_experiment_data:
        metrics = exp_data["noise_robustness"][noise]["metrics"]
        losses = exp_data["noise_robustness"][noise]["losses"]
        train_metrics.append(metrics["train"])
        val_metrics.append(metrics["val"])
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])

# Convert to numpy array for easier calculations
train_metrics = np.array(train_metrics)
val_metrics = np.array(val_metrics)
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

# Calculate mean and standard error
mean_train_metrics = np.mean(train_metrics, axis=0)
mean_val_metrics = np.mean(val_metrics, axis=0)
std_error_train_metrics = np.std(train_metrics, axis=0) / np.sqrt(
    train_metrics.shape[0]
)
std_error_val_metrics = np.std(val_metrics, axis=0) / np.sqrt(val_metrics.shape[0])

mean_train_losses = np.mean(train_losses, axis=0)
mean_val_losses = np.mean(val_losses, axis=0)
std_error_train_losses = np.std(train_losses, axis=0) / np.sqrt(train_losses.shape[0])
std_error_val_losses = np.std(val_losses, axis=0) / np.sqrt(val_losses.shape[0])

try:
    plt.figure()
    epochs = np.arange(len(mean_train_metrics))
    plt.plot(epochs, mean_train_metrics, label="Mean Train Accuracy")
    plt.plot(epochs, mean_val_metrics, label="Mean Validation Accuracy")
    plt.fill_between(
        epochs,
        mean_train_metrics - std_error_train_metrics,
        mean_train_metrics + std_error_train_metrics,
        alpha=0.2,
    )
    plt.fill_between(
        epochs,
        mean_val_metrics - std_error_val_metrics,
        mean_val_metrics + std_error_val_metrics,
        alpha=0.2,
    )
    plt.title("Aggregated Results - Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(epochs, mean_train_losses, label="Mean Train Loss")
    plt.plot(epochs, mean_val_losses, label="Mean Validation Loss")
    plt.fill_between(
        epochs,
        mean_train_losses - std_error_train_losses,
        mean_train_losses + std_error_train_losses,
        alpha=0.2,
    )
    plt.fill_between(
        epochs,
        mean_val_losses - std_error_val_losses,
        mean_val_losses + std_error_val_losses,
        alpha=0.2,
    )
    plt.title("Aggregated Results - Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()
