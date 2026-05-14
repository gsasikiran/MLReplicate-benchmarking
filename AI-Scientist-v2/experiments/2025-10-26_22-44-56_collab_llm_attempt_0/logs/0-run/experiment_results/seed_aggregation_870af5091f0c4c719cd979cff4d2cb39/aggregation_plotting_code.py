import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_ca7b0f50a5884fb9abe86a579f0086ea_proc_2540027/experiment_data.npy",
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_f8f0bc172eba41f7bd9fbbf44429e049_proc_2540030/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Compute the mean and standard error for Training Losses
try:
    losses = []
    for experiment in all_experiment_data:
        losses.append(
            experiment["momentum_tuning"]["synthetic_dataset"]["losses"]["train"]
        )
    means = np.mean(losses, axis=0)
    std_errors = np.std(losses, axis=0) / np.sqrt(len(losses))

    plt.figure()
    plt.errorbar(
        np.arange(len(means)), means, yerr=std_errors, label="Mean Training Loss ± SE"
    )
    plt.title("Mean Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")
    plt.close()

# Compute the mean and standard error for UES Metrics
try:
    metrics = []
    for experiment in all_experiment_data:
        metrics.append(
            experiment["momentum_tuning"]["synthetic_dataset"]["metrics"]["train"]
        )
    means = np.mean(metrics, axis=0)
    std_errors = np.std(metrics, axis=0) / np.sqrt(len(metrics))

    plt.figure()
    plt.errorbar(
        np.arange(len(means)), means, yerr=std_errors, label="Mean UES Metric ± SE"
    )
    plt.title("Mean UES Metric Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("UES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_ues_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean UES metric plot: {e}")
    plt.close()
