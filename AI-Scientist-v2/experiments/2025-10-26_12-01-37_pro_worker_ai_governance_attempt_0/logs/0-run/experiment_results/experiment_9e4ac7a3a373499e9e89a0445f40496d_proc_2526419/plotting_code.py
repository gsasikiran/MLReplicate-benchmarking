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

try:
    plt.figure()
    plt.plot(
        experiment_data["learning_rate_variation"]["fixed_lr"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["learning_rate_variation"]["fixed_lr"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curve - Fixed Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Loss_Curve_Fixed_LR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Loss Curve (Fixed LR): {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["learning_rate_variation"]["scheduler_lr"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["learning_rate_variation"]["scheduler_lr"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curve - Scheduler Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Loss_Curve_Scheduler_LR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Loss Curve (Scheduler LR): {e}")
    plt.close()
