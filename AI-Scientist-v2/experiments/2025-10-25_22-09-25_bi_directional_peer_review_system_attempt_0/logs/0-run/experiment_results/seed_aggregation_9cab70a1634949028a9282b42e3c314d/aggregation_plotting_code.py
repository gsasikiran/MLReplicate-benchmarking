import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_8dc2c8afb3a84b59b155d160528e7b18_proc_2517256/experiment_data.npy",
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_1fc1225f96a3452abe444879c0e5fa2f_proc_2517258/experiment_data.npy",
        "experiments/2025-10-25_22-09-25_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_20b61921a4284a0dab4f16e2d1c8e73d_proc_2517255/experiment_data.npy",
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

to_plot = ["synthetic_data", "train"]
for metric in ["losses", "metrics"]:
    try:
        plt.figure()
        data_collection = []
        for experiment_data in all_experiment_data:
            data = experiment_data["hyperparam_tuning_num_hidden_units"][to_plot[0]][
                metric
            ][to_plot[1]]
            data_collection.append(data)

        # Calculate mean and standard error
        mean_values = np.mean(data_collection, axis=0)
        std_error_values = np.std(data_collection, axis=0) / np.sqrt(
            len(data_collection)
        )

        epochs = range(len(mean_values))
        plt.plot(epochs, mean_values, label="Mean")
        plt.fill_between(
            epochs,
            mean_values - std_error_values,
            mean_values + std_error_values,
            alpha=0.1,
            label="Standard Error",
        )
        plt.title(f"{to_plot[0].capitalize()} {metric.capitalize()} over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{to_plot[0]}_{metric}_aggregated.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {metric} plot: {e}")
        plt.close()
