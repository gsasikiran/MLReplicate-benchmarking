import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses for each regularization type
for reg_type in experiment_data["regularization"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["regularization"][reg_type]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["regularization"][reg_type]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{reg_type.replace('_', ' ').title()} - Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{reg_type}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {reg_type} loss plot: {e}")
        plt.close()

# Plot validation metrics (PWIS) for each regularization type
for reg_type in experiment_data["regularization"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["regularization"][reg_type]["metrics"]["val"], label="PWIS"
        )
        plt.title(f"{reg_type.replace('_', ' ').title()} - Validation Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{reg_type}_validation_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {reg_type} validation metric plot: {e}")
        plt.close()
