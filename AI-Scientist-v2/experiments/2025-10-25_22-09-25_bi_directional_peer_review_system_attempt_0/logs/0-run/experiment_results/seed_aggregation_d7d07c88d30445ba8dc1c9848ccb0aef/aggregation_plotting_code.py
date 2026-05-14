import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_f75fe656859a4c10b95f65557987cf01_proc_2516354/experiment_data.npy",
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_7923e3a8faba41fc9813865c9a009a4c_proc_2516352/experiment_data.npy",
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_0217f4c514c3403a9fdeef8512a2d000_proc_2516351/experiment_data.npy",
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

for dataset in all_experiment_data:
    try:
        train_losses = dataset["hyperparam_tuning_num_hidden_units"]["synthetic_data"][
            "losses"
        ]["train"]
        epochs = np.arange(len(train_losses))
        mean_loss = np.mean(train_losses)
        se_loss = np.std(train_losses) / np.sqrt(len(train_losses))

        plt.figure()
        plt.errorbar(
            epochs, train_losses, yerr=se_loss, label="Mean Loss ± SE", fmt="-o"
        )
        plt.title("Training Loss over Epochs - Synthetic Data")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    try:
        train_metrics = dataset["hyperparam_tuning_num_hidden_units"]["synthetic_data"][
            "metrics"
        ]["train"]
        mean_metric = np.mean(train_metrics)
        se_metric = np.std(train_metrics) / np.sqrt(len(train_metrics))

        plt.figure()
        plt.errorbar(
            epochs, train_metrics, yerr=se_metric, label="Mean RQS ± SE", fmt="-o"
        )
        plt.title("Training RQS Metric over Epochs - Synthetic Data")
        plt.xlabel("Epochs")
        plt.ylabel("RQS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_data_training_rqs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training RQS plot: {e}")
        plt.close()
