import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for func_name in experiment_data["hyperparam_tuning_activation"]:
    try:
        train_losses = experiment_data["hyperparam_tuning_activation"][func_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["hyperparam_tuning_activation"][func_name][
            "losses"
        ]["val"]
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{func_name} - Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{func_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {func_name}: {e}")
        plt.close()

    try:
        metrics_train = experiment_data["hyperparam_tuning_activation"][func_name][
            "metrics"
        ]["train"]
        metrics_val = experiment_data["hyperparam_tuning_activation"][func_name][
            "metrics"
        ]["val"]
        plt.figure()
        plt.plot(metrics_train, label="Training Metric")
        plt.plot(metrics_val, label="Validation Metric")
        plt.title(f"{func_name} - Metric Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{func_name}_metric_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {func_name}: {e}")
        plt.close()
