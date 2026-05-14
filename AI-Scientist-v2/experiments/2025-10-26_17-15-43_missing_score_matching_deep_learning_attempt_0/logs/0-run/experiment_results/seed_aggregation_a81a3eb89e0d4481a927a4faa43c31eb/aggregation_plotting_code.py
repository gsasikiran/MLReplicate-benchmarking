markdown
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_paths = [
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_8202180701224c36814882408d715655_proc_2534218/experiment_data.npy",
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_5c1a2429df53439d81d7064017bbcb2d_proc_2534219/experiment_data.npy",
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_662ac91c2751452eb5a7cff0971c860d_proc_2534217/experiment_data.npy",
    ]
    all_experiment_data = []
    for path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Calculate means and standard errors for training and validation losses
    train_losses = [
        data["synthetic_dataset"]["losses"]["train"] for data in all_experiment_data
    ]
    val_losses = [
        data["synthetic_dataset"]["losses"]["val"] for data in all_experiment_data
    ]

    mean_train_losses = np.mean(train_losses, axis=0)
    mean_val_losses = np.mean(val_losses, axis=0)
    se_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
    se_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

    epochs = range(1, len(mean_train_losses) + 1)

    plt.figure()
    plt.plot(epochs, mean_train_losses, label="Mean Training Loss")
    plt.plot(epochs, mean_val_losses, label="Mean Validation Loss")
    plt.fill_between(
        epochs,
        mean_train_losses - se_train_losses,
        mean_train_losses + se_train_losses,
        alpha=0.2,
        label="Training Loss SE",
    )
    plt.fill_between(
        epochs,
        mean_val_losses - se_val_losses,
        mean_val_losses + se_val_losses,
        alpha=0.2,
        label="Validation Loss SE",
    )
    plt.title("Aggregate Loss Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_aggregate_loss_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves plot: {e}")
    plt.close()
