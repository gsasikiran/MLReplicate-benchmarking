markdown
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-10-26_13-24-43_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_67b1ff3e13054fd99eacba946aa89c17_proc_2529104/experiment_data.npy",
    "experiments/2025-10-26_13-24-43_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_c2eaecb99ca84fcbb1e4c7f0c33e92b0_proc_2529106/experiment_data.npy",
    "experiments/2025-10-26_13-24-43_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_e9bfa6f3292246c291808ab3be96dfcf_proc_2529105/experiment_data.npy",
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

# Aggregate losses
train_losses, val_losses = [], []
for data in all_experiment_data:
    train_losses.append(data["synthetic_dataset"]["losses"]["train"])
    val_losses.append(data["synthetic_dataset"]["losses"]["val"])

mean_train_loss = np.mean(train_losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
sem_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
sem_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

# Plot Loss Curves
try:
    epochs = range(1, len(mean_train_loss) + 1)
    plt.figure()
    plt.plot(epochs, mean_train_loss, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_train_loss - sem_train_loss,
        mean_train_loss + sem_train_loss,
        alpha=0.2,
    )
    plt.plot(epochs, mean_val_loss, label="Mean Validation Loss")
    plt.fill_between(
        epochs, mean_val_loss - sem_val_loss, mean_val_loss + sem_val_loss, alpha=0.2
    )
    plt.title("Loss Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss curve plot: {e}")
    plt.close()

# Plot Ground Truth vs Predictions
try:
    all_ground_truth = np.concatenate(
        [data["synthetic_dataset"]["ground_truth"] for data in all_experiment_data]
    )
    all_predictions = np.concatenate(
        [data["synthetic_dataset"]["predictions"] for data in all_experiment_data]
    )
    plt.figure()
    plt.scatter(all_ground_truth, all_predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], "r--", label="Ideal Line")
    plt.title("Ground Truth vs Predictions for Synthetic Dataset")
    plt.xlabel("Ground Truth Satisfaction")
    plt.ylabel("Predicted Satisfaction")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_dataset_ground_truth_vs_predictions_aggregate.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
