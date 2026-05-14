import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_0e394de3a39c48348f73df5f0a75da14_proc_2532099/experiment_data.npy",
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_775e1ffb89ef4fa5bd3660147890f85a_proc_2532102/experiment_data.npy",
        "experiments/2025-10-26_15-25-31_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_63d44fbf722a4b0bafe1c85f9845f514_proc_2532101/experiment_data.npy",
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

metric_names = ["losses", "metrics"]
for metric in metric_names:
    try:
        combined_train = []
        combined_val = []
        for exp_data in all_experiment_data:
            combined_train.append(
                exp_data["activation_function_ablation"]["synthetic_data"][metric][
                    "train"
                ]
            )
            combined_val.append(
                exp_data["activation_function_ablation"]["synthetic_data"][metric][
                    "val"
                ]
            )

        train_mean = np.mean(combined_train, axis=0)
        train_se = np.std(combined_train, axis=0) / np.sqrt(len(combined_train))
        val_mean = np.mean(combined_val, axis=0)
        val_se = np.std(combined_val, axis=0) / np.sqrt(len(combined_val))

        epochs = range(
            len(train_mean)
        )  # Assuming all experiments have same number of epochs

        plt.figure()
        plt.plot(epochs, train_mean, label="Mean Training", color="blue")
        plt.fill_between(
            epochs,
            train_mean - train_se,
            train_mean + train_se,
            color="blue",
            alpha=0.1,
            label="Training SE",
        )
        plt.plot(epochs, val_mean, label="Mean Validation", color="orange")
        plt.fill_between(
            epochs,
            val_mean - val_se,
            val_mean + val_se,
            color="orange",
            alpha=0.1,
            label="Validation SE",
        )
        plt.title(f"Mean and SE of Training and Validation {metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"synthetic_data_{metric}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {metric}: {e}")
        plt.close()
