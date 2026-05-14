import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_411c009f84424b8f8d72bb66980b7210_proc_2525844/experiment_data.npy",
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_5cee8600bcd045f1a04430117e37a8b5_proc_2525843/experiment_data.npy",
        "experiments/2025-10-26_12-01-37_pro_worker_ai_governance_attempt_0/logs/0-run/experiment_results/experiment_f76533053b61420bb0989fc86abe80e1_proc_2525846/experiment_data.npy",
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

# Calculate means and standard errors
mean_train_loss = np.mean(train_losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
stderr_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
stderr_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

try:
    plt.figure()
    plt.plot(mean_train_loss, label="Mean Train Loss")
    plt.fill_between(
        range(len(mean_train_loss)),
        mean_train_loss - stderr_train_loss,
        mean_train_loss + stderr_train_loss,
        alpha=0.2,
    )
    plt.plot(mean_val_loss, label="Mean Validation Loss")
    plt.fill_between(
        range(len(mean_val_loss)),
        mean_val_loss - stderr_val_loss,
        mean_val_loss + stderr_val_loss,
        alpha=0.2,
    )
    plt.title("Mean Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_training_vs_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss plot: {e}")
    plt.close()

try:
    plt.figure()
    for hidden_units in hidden_units_list:
        plt.scatter(
            experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                "ground_truth"
            ],
            experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                "predictions"
            ],
            alpha=0.5,
            label=f"Hidden Units: {hidden_units}",
        )
    plt.plot([0, 1], [0, 1], "r--")  # Line for reference
    plt.title("Ground Truth vs Predictions Across Hidden Units")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ground_truth_vs_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating prediction plot: {e}")
    plt.close()
