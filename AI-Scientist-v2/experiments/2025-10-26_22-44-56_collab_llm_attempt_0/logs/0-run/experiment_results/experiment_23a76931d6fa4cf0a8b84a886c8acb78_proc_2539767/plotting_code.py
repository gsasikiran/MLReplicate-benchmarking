import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training losses
try:
    learning_rates = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_dataset"
    ]["losses"]["train"]
    plt.figure()
    for lr_idx, lr in enumerate([0.0001, 0.001, 0.01]):
        plt.plot(
            range(1, len(learning_rates[lr_idx]) + 1),
            learning_rates[lr_idx],
            label=f"LR: {lr}",
        )
    plt.title("Training Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot UES metrics
try:
    ues_metrics = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_dataset"
    ]["metrics"]["train"]
    plt.figure()
    for lr_idx, lr in enumerate([0.0001, 0.001, 0.01]):
        plt.plot(
            range(1, len(ues_metrics[lr_idx]) + 1),
            ues_metrics[lr_idx],
            label=f"LR: {lr}",
        )
    plt.title("UES Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("UES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_ues_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating UES metrics plot: {e}")
    plt.close()
