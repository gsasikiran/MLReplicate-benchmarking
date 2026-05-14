import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_7937fb4a17c346128a8c71658dbf2325_proc_2518489/experiment_data.npy",
        "experiments/2025-10-25_23-33-50_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_f58839fa79d94a01a514a388ba4bb4cb_proc_2518491/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)

    # Extract training and validation losses
    train_losses = [
        data["peer_review"]["losses"]["train"] for data in all_experiment_data
    ]
    val_losses = [data["peer_review"]["losses"]["val"] for data in all_experiment_data]

    # Calculate mean and standard error
    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    se_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
    se_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    try:
        plt.figure()
        epochs = np.arange(len(mean_train_loss))
        plt.plot(epochs, mean_train_loss, label="Mean Training Loss", color="blue")
        plt.fill_between(
            epochs,
            mean_train_loss - se_train_loss,
            mean_train_loss + se_train_loss,
            color="blue",
            alpha=0.2,
            label="SE Training Loss",
        )
        plt.plot(epochs, mean_val_loss, label="Mean Validation Loss", color="orange")
        plt.fill_between(
            epochs,
            mean_val_loss - se_val_loss,
            mean_val_loss + se_val_loss,
            color="orange",
            alpha=0.2,
            label="SE Validation Loss",
        )
        plt.title("Loss Curves for Peer Review Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "peer_review_loss_curves_with_error_bars.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training/validation loss plot: {e}")
        plt.close()
except Exception as e:
    print(f"Error loading experiment data: {e}")
