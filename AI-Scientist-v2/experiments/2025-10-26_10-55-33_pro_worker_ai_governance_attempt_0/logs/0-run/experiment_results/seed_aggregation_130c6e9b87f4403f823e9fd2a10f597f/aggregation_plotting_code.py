import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
experiment_data_path_list = [
    "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_0c92318442564488b1e0b39fad891c42_proc_2523675/experiment_data.npy",
    "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_dc2aa8a77181457dbf1790aad1792739_proc_2523678/experiment_data.npy",
    "experiments/2025-10-26_10-55-33_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_aa9bb7e4fe2341139fb0723026252bec_proc_2523677/experiment_data.npy",
]

all_losses_train = []
all_losses_val = []
all_metrics_val = []

for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_losses_train.append(experiment_data["synthetic_dataset"]["losses"]["train"])
        all_losses_val.append(experiment_data["synthetic_dataset"]["losses"]["val"])
        all_metrics_val.append(experiment_data["synthetic_dataset"]["metrics"]["val"])
    except Exception as e:
        print(f"Error loading experiment data: {e}")

# Aggregate results
mean_loss_train = np.mean(all_losses_train, axis=0)
mean_loss_val = np.mean(all_losses_val, axis=0)
mean_metrics_val = np.mean(all_metrics_val, axis=0)
se_loss_train = np.std(all_losses_train, axis=0) / np.sqrt(len(all_losses_train))
se_loss_val = np.std(all_losses_val, axis=0) / np.sqrt(len(all_losses_val))
se_metrics_val = np.std(all_metrics_val, axis=0) / np.sqrt(len(all_metrics_val))

# Plot training and validation losses
try:
    plt.figure()
    plt.plot(mean_loss_train, label="Mean Train Loss")
    plt.plot(mean_loss_val, label="Mean Validation Loss")
    plt.fill_between(
        range(len(mean_loss_train)),
        mean_loss_train - se_loss_train,
        mean_loss_train + se_loss_train,
        color="b",
        alpha=0.1,
    )
    plt.fill_between(
        range(len(mean_loss_val)),
        mean_loss_val - se_loss_val,
        mean_loss_val + se_loss_val,
        color="orange",
        alpha=0.1,
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_dataset_training_validation_losses_aggregated.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

# Plot validation performance (PWIS)
try:
    plt.figure()
    plt.plot(mean_metrics_val, label="Mean PWIS")
    plt.fill_between(
        range(len(mean_metrics_val)),
        mean_metrics_val - se_metrics_val,
        mean_metrics_val + se_metrics_val,
        color="g",
        alpha=0.1,
    )
    plt.title("Validation Metric (PWIS) Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_dataset_validation_metric_pw_aggregated.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
