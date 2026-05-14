import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_paths = [
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_e2d3fa6fa8e747298767aeea4a5a9bed_proc_2519528/experiment_data.npy",
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_56270b8ccd464b52a8e61ac20e6ae787_proc_2519527/experiment_data.npy",
    "experiments/2025-10-26_00-53-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_ba43def4f2fa4831b31459f256f56349_proc_2519529/experiment_data.npy",
]

all_experiment_data = []
for experiment_data_path in experiment_data_paths:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data: {e}")

# Aggregating metrics
train_losses = np.array(
    [data["peer_review"]["losses"]["train"] for data in all_experiment_data]
)
val_losses = np.array(
    [data["peer_review"]["losses"]["val"] for data in all_experiment_data]
)
train_rqs = np.array(
    [data["peer_review"]["metrics"]["train"] for data in all_experiment_data]
)
val_rqs = np.array(
    [data["peer_review"]["metrics"]["val"] for data in all_experiment_data]
)

# Calculate mean and standard error
mean_train_loss = np.mean(train_losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
se_train_loss = np.std(train_losses, axis=0) / np.sqrt(train_losses.shape[0])
se_val_loss = np.std(val_losses, axis=0) / np.sqrt(val_losses.shape[0])
mean_train_rqs = np.mean(train_rqs, axis=0)
mean_val_rqs = np.mean(val_rqs, axis=0)
se_train_rqs = np.std(train_rqs, axis=0) / np.sqrt(train_rqs.shape[0])
se_val_rqs = np.std(val_rqs, axis=0) / np.sqrt(val_rqs.shape[0])

# Plot Training and Validation Losses
try:
    epochs = range(1, len(mean_train_loss) + 1)
    plt.figure()
    plt.plot(epochs, mean_train_loss, label="Mean Train Loss")
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
    plt.title("Mean Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss plot: {e}")
    plt.close()

# Plot Training and Validation RQS Metrics
try:
    plt.figure()
    plt.plot(epochs, mean_train_rqs, label="Mean Train RQS")
    plt.plot(epochs, mean_val_rqs, label="Mean Validation RQS")
    plt.fill_between(
        epochs, mean_train_rqs - se_train_rqs, mean_train_rqs + se_train_rqs, alpha=0.1
    )
    plt.fill_between(
        epochs, mean_val_rqs - se_val_rqs, mean_val_rqs + se_val_rqs, alpha=0.1
    )
    plt.title("Mean Training and Validation RQS Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_training_validation_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean RQS plot: {e}")
    plt.close()
