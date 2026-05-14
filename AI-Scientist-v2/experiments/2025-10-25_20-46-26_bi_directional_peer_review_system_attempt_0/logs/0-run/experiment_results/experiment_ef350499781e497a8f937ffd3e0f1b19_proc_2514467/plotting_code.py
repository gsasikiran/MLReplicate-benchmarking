import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    plt.figure()
    for lr, (train_losses, val_losses) in zip(
        experiment_data["hyperparam_tuning"]["learning_rate"].keys(),
        zip(
            experiment_data["hyperparam_tuning"]["learning_rate"]["losses"]["train"],
            experiment_data["hyperparam_tuning"]["learning_rate"]["losses"]["val"],
        ),
    ):
        plt.plot(range(1, len(train_losses) + 1), train_losses, label=f"Train LR={lr}")
        plt.plot(
            range(1, len(val_losses) + 1),
            val_losses,
            linestyle="dashed",
            label=f"Val LR={lr}",
        )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    metrics = experiment_data["hyperparam_tuning"]["learning_rate"]["metrics"]["train"]
    plt.plot(metrics, marker="o")
    plt.title("Training Metrics Over Different Learning Rates")
    plt.xlabel("Learning Rate Index")
    plt.ylabel("Metric Value")
    plt.xticks(
        range(len(metrics)),
        list(experiment_data["hyperparam_tuning"]["learning_rate"].keys()),
    )
    plt.savefig(os.path.join(working_dir, "training_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
