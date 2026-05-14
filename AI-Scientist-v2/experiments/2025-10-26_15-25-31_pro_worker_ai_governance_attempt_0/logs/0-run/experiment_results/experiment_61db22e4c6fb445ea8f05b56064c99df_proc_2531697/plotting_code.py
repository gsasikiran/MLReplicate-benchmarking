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

learning_rates = experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"][
    "learning_rates"
]
train_losses = experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"][
    "losses"
]["train"]
val_losses = experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"][
    "losses"
]["val"]
train_metrics = experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"][
    "metrics"
]["train"]
val_metrics = experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"][
    "metrics"
]["val"]

try:
    plt.figure()
    for i, lr in enumerate(learning_rates):
        plt.plot(train_losses[i], label=f"Train Loss (lr={lr})")
        plt.plot(val_losses[i], label=f"Validation Loss (lr={lr})", linestyle="--")
    plt.title("Loss Curves for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    for i, lr in enumerate(learning_rates):
        plt.plot(train_metrics[i], label=f"Train Accuracy (lr={lr})")
        plt.plot(val_metrics[i], label=f"Validation Accuracy (lr={lr})", linestyle="--")
    plt.title("Accuracy Curves for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()
