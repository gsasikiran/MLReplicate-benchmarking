import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_377de229138e4eb7a7f1d67049d66f13_proc_2518155/experiment_data.npy",
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_b7d0cba006414613a274a8f62da5ec40_proc_2518154/experiment_data.npy",
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_dcd6cf28664e4d54b973b45823856747_proc_2518156/experiment_data.npy",
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
    # Aggregating training and validation losses
    train_losses = []
    val_losses = []
    for experiment_data in all_experiment_data:
        train_losses.append(experiment_data["peer_review"]["losses"]["train"])
        val_losses.append(experiment_data["peer_review"]["losses"]["val"])

    # Calculate mean and standard error for losses
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    se_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
    se_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    # Plotting aggregated training and validation losses
    plt.figure()
    epochs = np.arange(len(mean_train_loss))
    plt.plot(epochs, mean_train_loss, label="Mean Training Loss")
    plt.plot(epochs, mean_val_loss, label="Mean Validation Loss")
    plt.fill_between(
        epochs,
        mean_train_loss - se_train_loss,
        mean_train_loss + se_train_loss,
        alpha=0.1,
        color="blue",
    )
    plt.fill_between(
        epochs,
        mean_val_loss - se_val_loss,
        mean_val_loss + se_val_loss,
        alpha=0.1,
        color="orange",
    )
    plt.title("Aggregated Loss Curves for Peer Review Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_aggregated_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training/validation loss plot: {e}")
    plt.close()
