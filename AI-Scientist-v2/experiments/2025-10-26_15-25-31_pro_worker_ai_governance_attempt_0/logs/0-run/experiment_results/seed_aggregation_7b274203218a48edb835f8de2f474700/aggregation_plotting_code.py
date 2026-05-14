import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_cae42f8d72d146b69c0f783f9d9b614d_proc_2531092/experiment_data.npy",
    "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_f2a94b0a4b93439d98a3fc08c6852a83_proc_2531089/experiment_data.npy",
    "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_3fc0bebe3cff4b67ace6ad117a5d23cf_proc_2531091/experiment_data.npy",
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


# Calculate mean and standard error for losses and metrics
def calculate_mean_std(data):
    return np.mean(data, axis=0), np.std(data, axis=0) / np.sqrt(data.shape[0])


try:
    plt.figure()
    losses_train = np.array(
        [
            exp["synthetic_data"]["losses"]["train"]
            for exp in all_experiment_data
            if "losses" in exp["synthetic_data"]
        ]
    )
    losses_val = np.array(
        [
            exp["synthetic_data"]["losses"]["val"]
            for exp in all_experiment_data
            if "losses" in exp["synthetic_data"]
        ]
    )

    mean_train_loss, se_train_loss = calculate_mean_std(losses_train)
    mean_val_loss, se_val_loss = calculate_mean_std(losses_val)

    plt.plot(mean_train_loss, label="Mean Train Loss")
    plt.fill_between(
        range(len(mean_train_loss)),
        mean_train_loss - se_train_loss,
        mean_train_loss + se_train_loss,
        alpha=0.2,
    )
    plt.plot(mean_val_loss, label="Mean Validation Loss")
    plt.fill_between(
        range(len(mean_val_loss)),
        mean_val_loss - se_val_loss,
        mean_val_loss + se_val_loss,
        alpha=0.2,
    )

    plt.title("Mean Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_mean_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss plot: {e}")

try:
    plt.figure()
    metrics_train = np.array(
        [
            exp["synthetic_data"]["metrics"]["train"]
            for exp in all_experiment_data
            if "metrics" in exp["synthetic_data"]
        ]
    )
    metrics_val = np.array(
        [
            exp["synthetic_data"]["metrics"]["val"]
            for exp in all_experiment_data
            if "metrics" in exp["synthetic_data"]
        ]
    )

    mean_train_acc, se_train_acc = calculate_mean_std(metrics_train)
    mean_val_acc, se_val_acc = calculate_mean_std(metrics_val)

    plt.plot(mean_train_acc, label="Mean Train Accuracy")
    plt.fill_between(
        range(len(mean_train_acc)),
        mean_train_acc - se_train_acc,
        mean_train_acc + se_train_acc,
        alpha=0.2,
    )
    plt.plot(mean_val_acc, label="Mean Validation Accuracy")
    plt.fill_between(
        range(len(mean_val_acc)),
        mean_val_acc - se_val_acc,
        mean_val_acc + se_val_acc,
        alpha=0.2,
    )

    plt.title("Mean Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_mean_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean accuracy plot: {e}")
