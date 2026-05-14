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

for ablation in [
    "ablation_job_displacement",
    "ablation_wage_change",
    "ablation_retraining_access",
]:
    try:
        val_losses = experiment_data[ablation]["synthetic_worker_data"]["losses"]["val"]
        plt.figure()
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{ablation}: Validation Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ablation}_validation_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot for {ablation}: {e}")
        plt.close()

    try:
        PWIS = experiment_data[ablation]["synthetic_worker_data"]["PWIS"]
        plt.figure()
        plt.plot(PWIS, label="PWIS")
        plt.title(f"{ablation}: PWIS Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ablation}_PWIS.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PWIS plot for {ablation}: {e}")
        plt.close()

    try:
        predictions = experiment_data[ablation]["synthetic_worker_data"]["predictions"]
        ground_truth = experiment_data[ablation]["synthetic_worker_data"][
            "ground_truth"
        ]
        plt.figure()
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.title(f"{ablation}: Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"{ablation}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions vs ground truth plot for {ablation}: {e}")
        plt.close()
