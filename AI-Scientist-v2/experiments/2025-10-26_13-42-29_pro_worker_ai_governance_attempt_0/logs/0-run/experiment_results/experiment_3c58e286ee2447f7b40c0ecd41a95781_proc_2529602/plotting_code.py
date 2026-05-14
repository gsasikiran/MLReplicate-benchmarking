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

try:
    # Plot training loss
    plt.figure()
    plt.plot(
        experiment_data["early_stopping"]["synthetic_dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["early_stopping"]["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot validation metrics
    plt.figure()
    plt.plot(
        experiment_data["early_stopping"]["synthetic_dataset"]["metrics"]["val"],
        label="Validation WWBI",
    )
    plt.title("Validation WWBI for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("WWBI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_validation_wwbi.png"))
    plt.close()
except Exception as e:
    print(f"Error creating WWBI metric plot: {e}")
    plt.close()
