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

for lr in [0.001, 0.01, 0.1]:
    try:
        plt.figure()
        train_losses = experiment_data[f"hyperparam_tuning_lr_{lr}"]["synthetic_data"][
            "losses"
        ]["train"]
        plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
        plt.title(f"Training Loss Curve - Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for LR {lr}: {e}")
        plt.close()

    try:
        plt.figure()
        rqs = experiment_data[f"hyperparam_tuning_lr_{lr}"]["synthetic_data"][
            "metrics"
        ]["train"]
        plt.plot(range(len(rqs)), rqs, label="Training RQS", color="orange")
        plt.title(f"Training RQS Curve - Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("RQS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_rqs_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RQS plot for LR {lr}: {e}")
        plt.close()
