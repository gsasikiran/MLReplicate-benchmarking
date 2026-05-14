markdown
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_bf39a9b6106c45ae90868c130ae0267f_proc_2526043/experiment_data.npy",
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_db3392f375e7454b8a946cf6b79ec02a_proc_2526042/experiment_data.npy",
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_e13f1cce90db48e29e2b0bfa68b79e44_proc_2526044/experiment_data.npy",
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

hidden_units_list = experiment_data["hyperparam_tuning"]["hidden_units"]
train_losses = []
val_losses = []

for hidden_units in hidden_units_list:
    train_losses.append(
        experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units]["losses"][
            "train"
        ]
    )
    val_losses.append(
        experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units]["losses"][
            "val"
        ]
    )

mean_train_loss = np.mean(train_losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
sem_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(hidden_units_list))
sem_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(hidden_units_list))

try:
    plt.figure()
    epochs = np.arange(len(mean_train_loss))
    plt.plot(epochs, mean_train_loss, label="Mean Train Loss")
    plt.fill_between(
        epochs,
        mean_train_loss - sem_train_loss,
        mean_train_loss + sem_train_loss,
        alpha=0.2,
    )
    plt.plot(epochs, mean_val_loss, label="Mean Validation Loss")
    plt.fill_between(
        epochs, mean_val_loss - sem_val_loss, mean_val_loss + sem_val_loss, alpha=0.2
    )
    plt.title("Mean Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"mean_training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss plot: {e}")
    plt.close()

try:
    # Scatter plot for ground truth vs predictions aggregation would go here
    pass  # Replace with code similar to above if necessary
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
