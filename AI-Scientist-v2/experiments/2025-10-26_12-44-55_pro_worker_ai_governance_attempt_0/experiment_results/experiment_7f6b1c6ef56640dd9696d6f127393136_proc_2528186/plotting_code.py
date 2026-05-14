import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
for feature, feature_name in zip(
    ["ablation_job_displacement", "ablation_wage_change", "ablation_retraining_access"],
    ["Job Displacement", "Wage Change", "Retraining Access"],
):
    try:
        plt.figure()
        plt.plot(
            experiment_data[feature]["synthetic_worker_data"]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data[feature]["synthetic_worker_data"]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{feature_name} - Training vs Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{feature_name}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {feature_name} losses: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data[feature]["synthetic_worker_data"]["metrics"]["val"],
            label="WIS",
            marker="o",
        )
        plt.title(f"{feature_name} - Validation Metrics (WIS)")
        plt.xlabel("Epochs")
        plt.ylabel("WIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{feature_name}_WIS.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {feature_name} WIS: {e}")
        plt.close()
