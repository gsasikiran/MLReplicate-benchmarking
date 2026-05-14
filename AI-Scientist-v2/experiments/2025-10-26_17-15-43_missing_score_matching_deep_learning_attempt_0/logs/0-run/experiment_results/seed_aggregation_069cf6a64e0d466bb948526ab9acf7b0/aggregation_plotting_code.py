import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_3a3a95e6fa714a82b2d97d20aab40767_proc_2534398/experiment_data.npy",
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_30c9cb82519548ecb3519cc267aa87f8_proc_2534399/experiment_data.npy",
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_44f79a08b9934faf83516a80bd738c6e_proc_2534396/experiment_data.npy",
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

for activation, data in all_experiment_data[0][
    "hyperparam_tuning_activation_function"
].items():
    try:
        epochs = np.arange(len(data["losses"]["train"]))
        mean_train_loss = np.mean(data["losses"]["train"])
        mean_val_loss = np.mean(data["losses"]["val"])
        stderr_train_loss = np.std(data["losses"]["train"]) / np.sqrt(
            len(data["losses"]["train"])
        )
        stderr_val_loss = np.std(data["losses"]["val"]) / np.sqrt(
            len(data["losses"]["val"])
        )

        plt.figure()
        plt.errorbar(
            epochs,
            np.repeat(mean_train_loss, len(epochs)),
            yerr=stderr_train_loss,
            label="Training Loss (mean ± SE)",
        )
        plt.errorbar(
            epochs,
            np.repeat(mean_val_loss, len(epochs)),
            yerr=stderr_val_loss,
            label="Validation Loss (mean ± SE)",
        )
        plt.title(f"{activation} Activation Function - Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {activation}: {e}")
        plt.close()

    try:
        mean_train_metric = np.mean(data["metrics"]["train"])
        mean_val_metric = np.mean(data["metrics"]["val"])
        stderr_train_metric = np.std(data["metrics"]["train"]) / np.sqrt(
            len(data["metrics"]["train"])
        )
        stderr_val_metric = np.std(data["metrics"]["val"]) / np.sqrt(
            len(data["metrics"]["val"])
        )

        plt.figure()
        plt.errorbar(
            epochs,
            np.repeat(mean_train_metric, len(epochs)),
            yerr=stderr_train_metric,
            label="Training Metric (mean ± SE)",
        )
        plt.errorbar(
            epochs,
            np.repeat(mean_val_metric, len(epochs)),
            yerr=stderr_val_metric,
            label="Validation Metric (mean ± SE)",
        )
        plt.title(f"{activation} Activation Function - Metrics Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation}_metric_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {activation}: {e}")
        plt.close()
