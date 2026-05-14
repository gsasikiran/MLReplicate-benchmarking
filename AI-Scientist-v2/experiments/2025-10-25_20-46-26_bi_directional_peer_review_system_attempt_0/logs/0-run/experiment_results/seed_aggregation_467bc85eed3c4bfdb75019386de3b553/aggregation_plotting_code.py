import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_8edd52608d944038871082d9eadf8d85_proc_2514779/experiment_data.npy",
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_4d233c8e67234c69a8910de59a6b4506_proc_2514777/experiment_data.npy",
        "experiments/2025-10-25_20-46-26_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_2be122161b00478293a2255eacbba034_proc_2514776/experiment_data.npy",
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

try:
    losses = []
    for experiment in all_experiment_data:
        train_losses = experiment["activation_function_tuning"]["FeedbackDataset"][
            "losses"
        ]["train"]
        val_losses = experiment["activation_function_tuning"]["FeedbackDataset"][
            "losses"
        ]["val"]
        losses.append((train_losses, val_losses))

    train_losses, val_losses = zip(*losses)

    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    std_train_loss = np.std(train_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)

    epochs = range(1, len(mean_train_loss) + 1)
    plt.figure()
    plt.plot(epochs, mean_train_loss, label="Mean Training Loss", color="blue")
    plt.fill_between(
        epochs,
        mean_train_loss - std_train_loss,
        mean_train_loss + std_train_loss,
        color="blue",
        alpha=0.2,
        label="Standard Error (Train)",
    )
    plt.plot(epochs, mean_val_loss, label="Mean Validation Loss", color="orange")
    plt.fill_between(
        epochs,
        mean_val_loss - std_val_loss,
        mean_val_loss + std_val_loss,
        color="orange",
        alpha=0.2,
        label="Standard Error (Validation)",
    )
    plt.title("Aggregated Loss Curves - Feedback Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_aggregated_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

try:
    metrics = []
    for experiment in all_experiment_data:
        train_metrics = experiment["activation_function_tuning"]["FeedbackDataset"][
            "metrics"
        ]["train"]
        metrics.append(train_metrics)

    mean_metrics = np.mean(metrics, axis=0)
    std_metrics = np.std(metrics, axis=0)

    plt.figure()
    epochs = range(1, len(mean_metrics) + 1)
    plt.plot(epochs, mean_metrics, label="Mean Training Metrics", color="green")
    plt.fill_between(
        epochs,
        mean_metrics - std_metrics,
        mean_metrics + std_metrics,
        color="green",
        alpha=0.2,
        label="Standard Error (Metrics)",
    )
    plt.title("Aggregated Training Metrics Over Epochs - Feedback Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics Value")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "FeedbackDataset_aggregated_training_metrics.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated metrics plot: {e}")
    plt.close()
