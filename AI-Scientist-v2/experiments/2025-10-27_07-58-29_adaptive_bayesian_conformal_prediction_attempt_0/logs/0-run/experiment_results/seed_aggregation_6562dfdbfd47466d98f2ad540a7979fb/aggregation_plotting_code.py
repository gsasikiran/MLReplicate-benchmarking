import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_fd2ebbd90f3e4e2cb365cc967faad135_proc_2544333/experiment_data.npy",
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_070dda08713b4bd789b33d69fc567e59_proc_2544332/experiment_data.npy",
        "experiments/2025-10-27_07-58-29_adaptive_bayesian_conformal_prediction_attempt_0/logs/0-run/experiment_results/experiment_c4f64867a88c47e2b9cf351dafedc9b2_proc_2544330/experiment_data.npy",
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

for scale_type in all_experiment_data[0]["feature_scale_investigation"]:
    # Aggregate losses for training and validation
    train_losses = [
        exp["feature_scale_investigation"][scale_type]["losses"]["train"]
        for exp in all_experiment_data
    ]
    val_losses = [
        exp["feature_scale_investigation"][scale_type]["losses"]["val"]
        for exp in all_experiment_data
    ]

    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    se_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(all_experiment_data))
    se_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(all_experiment_data))

    try:
        epochs = np.arange(len(mean_train_loss))
        plt.figure()
        plt.plot(epochs, mean_train_loss, label="Mean Train Loss", color="blue")
        plt.fill_between(
            epochs,
            mean_train_loss - se_train_loss,
            mean_train_loss + se_train_loss,
            color="blue",
            alpha=0.2,
            label="Train Loss SE",
        )
        plt.plot(epochs, mean_val_loss, label="Mean Validation Loss", color="orange")
        plt.fill_between(
            epochs,
            mean_val_loss - se_val_loss,
            mean_val_loss + se_val_loss,
            color="orange",
            alpha=0.2,
            label="Validation Loss SE",
        )
        plt.title(f"Aggregated Loss Curves for {scale_type} Scaling")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"aggregated_loss_curves_{scale_type}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve for {scale_type}: {e}")
        plt.close()

    # Aggregate validation reliability metrics
    val_metrics = [
        exp["feature_scale_investigation"][scale_type]["metrics"]["val"]
        for exp in all_experiment_data
    ]
    mean_val_metrics = np.mean(val_metrics, axis=0)
    se_val_metrics = np.std(val_metrics, axis=0) / np.sqrt(len(all_experiment_data))

    try:
        epochs = np.arange(len(mean_val_metrics))
        plt.figure()
        plt.plot(
            epochs,
            mean_val_metrics,
            label="Mean Validation Reliability",
            marker="o",
            color="green",
        )
        plt.fill_between(
            epochs,
            mean_val_metrics - se_val_metrics,
            mean_val_metrics + se_val_metrics,
            color="green",
            alpha=0.2,
            label="Validation Reliability SE",
        )
        plt.title(f"Aggregated Validation Reliability for {scale_type} Scaling")
        plt.xlabel("Epochs")
        plt.ylabel("Reliability Measure")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"aggregated_reliability_metric_{scale_type}.png")
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating aggregated reliability metric plot for {scale_type}: {e}"
        )
        plt.close()
