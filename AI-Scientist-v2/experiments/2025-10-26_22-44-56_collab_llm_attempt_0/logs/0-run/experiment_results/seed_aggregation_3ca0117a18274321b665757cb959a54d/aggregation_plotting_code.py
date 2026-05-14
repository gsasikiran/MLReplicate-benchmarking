import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "None/experiment_data.npy",
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_77c7eb192c10410581a4b94db211fd88_proc_2539770/experiment_data.npy",
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
    # Aggregate training losses
    losses_train = [
        exp["momentum_tuning"]["synthetic_dataset"]["losses"]["train"]
        for exp in all_experiment_data
    ]
    mean_losses = np.mean(losses_train, axis=0)
    se_losses = np.std(losses_train, axis=0) / np.sqrt(len(losses_train))

    plt.figure()
    plt.plot(mean_losses, label="Mean Training Loss")
    plt.fill_between(
        np.arange(len(mean_losses)),
        mean_losses - se_losses,
        mean_losses + se_losses,
        alpha=0.2,
        label="SE",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Aggregate UES metrics
    metrics_train = [
        exp["momentum_tuning"]["synthetic_dataset"]["metrics"]["train"]
        for exp in all_experiment_data
    ]
    mean_metrics = np.mean(metrics_train, axis=0)
    se_metrics = np.std(metrics_train, axis=0) / np.sqrt(len(metrics_train))

    plt.figure()
    plt.plot(mean_metrics, label="Mean UES Metric")
    plt.fill_between(
        np.arange(len(mean_metrics)),
        mean_metrics - se_metrics,
        mean_metrics + se_metrics,
        alpha=0.2,
        label="SE",
    )
    plt.title("UES Metric Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("UES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_ues_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating UES metric plot: {e}")
    plt.close()
