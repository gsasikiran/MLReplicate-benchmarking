import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_fb107a1455164a10ae75b8a6754ce118_proc_2529421/experiment_data.npy",
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_efea4ca1c85a4fe6a186c2303ed4240d_proc_2529422/experiment_data.npy",
        "experiments/2025-10-26_13-42-29_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_7b86409c297247d38b40738ce7676fff_proc_2529423/experiment_data.npy",
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

data_length = len(all_experiment_data)
try:
    epochs = np.arange(
        1, len(all_experiment_data[0]["synthetic_dataset"]["losses"]["train"]) + 1
    )

    train_losses = np.array(
        [exp["synthetic_dataset"]["losses"]["train"] for exp in all_experiment_data]
    )
    val_losses = np.array(
        [exp["synthetic_dataset"]["losses"]["val"] for exp in all_experiment_data]
    )
    train_metrics = np.array(
        [exp["synthetic_dataset"]["metrics"]["val"] for exp in all_experiment_data]
    )

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_val_losses = np.mean(val_losses, axis=0)
    se_train_losses = np.std(train_losses, axis=0) / np.sqrt(data_length)
    se_val_losses = np.std(val_losses, axis=0) / np.sqrt(data_length)

    plt.figure()
    plt.plot(epochs, mean_train_losses, label="Mean Train Loss", color="blue")
    plt.plot(epochs, mean_val_losses, label="Mean Validation Loss", color="orange")
    plt.fill_between(
        epochs,
        mean_train_losses - se_train_losses,
        mean_train_losses + se_train_losses,
        color="blue",
        alpha=0.2,
    )
    plt.fill_between(
        epochs,
        mean_val_losses - se_val_losses,
        mean_val_losses + se_val_losses,
        color="orange",
        alpha=0.2,
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Aggregated Training and Validation Loss Curves")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_dataset_aggregated_training_validation_loss.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve plot: {e}")
    plt.close()

try:
    mean_metrics = np.mean(train_metrics, axis=0)
    se_metrics = np.std(train_metrics, axis=0) / np.sqrt(data_length)

    plt.figure()
    plt.plot(epochs, mean_metrics, label="Mean WWBI Metric", color="green")
    plt.fill_between(
        epochs,
        mean_metrics - se_metrics,
        mean_metrics + se_metrics,
        color="green",
        alpha=0.2,
    )
    plt.xlabel("Epochs")
    plt.ylabel("WWBI")
    plt.title("Aggregated WWBI Metric Over Epochs")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_aggregated_wwbi_metric.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated WWBI metric plot: {e}")
    plt.close()
