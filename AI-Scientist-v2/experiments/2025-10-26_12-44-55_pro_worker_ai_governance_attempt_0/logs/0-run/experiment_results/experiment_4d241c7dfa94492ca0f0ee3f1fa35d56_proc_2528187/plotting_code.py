import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting averages for validation losses
try:
    plt.figure()
    plt.plot(
        experiment_data["ablation_job_displacement"]["synthetic_worker_data"]["losses"][
            "val"
        ],
        label="Job Displacement",
    )
    plt.plot(
        experiment_data["ablation_wage_change"]["synthetic_worker_data"]["losses"][
            "val"
        ],
        label="Wage Change",
    )
    plt.plot(
        experiment_data["ablation_retraining_access"]["synthetic_worker_data"][
            "losses"
        ]["val"],
        label="Retraining Access",
    )
    plt.title("Validation Losses by Feature Ablation")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation losses plot: {e}")
    plt.close()

# Plotting PWIS metrics
try:
    plt.figure()
    plt.plot(
        experiment_data["ablation_job_displacement"]["synthetic_worker_data"]["PWIS"],
        label="Job Displacement PWIS",
    )
    plt.plot(
        experiment_data["ablation_wage_change"]["synthetic_worker_data"]["PWIS"],
        label="Wage Change PWIS",
    )
    plt.plot(
        experiment_data["ablation_retraining_access"]["synthetic_worker_data"]["PWIS"],
        label="Retraining Access PWIS",
    )
    plt.title("PWIS Metrics by Feature Ablation")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "pwis_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS metrics plot: {e}")
    plt.close()

# Sample prediction vs ground truth plot for job displacement
try:
    plt.figure()
    plt.scatter(
        experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
            "ground_truth"
        ],
        experiment_data["ablation_job_displacement"]["synthetic_worker_data"][
            "predictions"
        ],
        alpha=0.5,
    )
    plt.title("Predictions vs Ground Truth - Job Displacement")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "predictions_vs_ground_truth_job_displacement.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot for job displacement: {e}")
    plt.close()
