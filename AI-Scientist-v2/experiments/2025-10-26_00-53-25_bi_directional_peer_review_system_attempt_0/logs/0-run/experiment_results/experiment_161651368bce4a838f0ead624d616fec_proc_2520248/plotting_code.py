import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for lr in experiment_data["hyperparam_tuning_lr"]:
    try:
        train_losses = experiment_data["hyperparam_tuning_lr"][lr]["losses"]["train"]
        val_losses = experiment_data["hyperparam_tuning_lr"][lr]["losses"]["val"]
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Learning Rate: {lr} - Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{lr}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for learning rate {lr}: {e}")
        plt.close()

    try:
        metrics_train = experiment_data["hyperparam_tuning_lr"][lr]["metrics"]["train"]
        metrics_val = experiment_data["hyperparam_tuning_lr"][lr]["metrics"]["val"]
        plt.figure()
        plt.plot(metrics_train, label="Training Metric")
        plt.plot(metrics_val, label="Validation Metric")
        plt.title(f"Learning Rate: {lr} - Metric Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{lr}_metric_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for learning rate {lr}: {e}")
        plt.close()
