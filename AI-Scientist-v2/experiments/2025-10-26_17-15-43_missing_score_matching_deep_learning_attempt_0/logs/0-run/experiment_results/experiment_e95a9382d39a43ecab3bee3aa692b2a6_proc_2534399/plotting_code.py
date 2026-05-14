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

# Plot training and validation losses
try:
    for dropout_rate in experiment_data["dropout_tuning"]["synthetic_dataset"][
        "losses"
    ].keys():
        plt.figure()
        plt.plot(
            experiment_data["dropout_tuning"]["synthetic_dataset"]["losses"][
                dropout_rate
            ]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["dropout_tuning"]["synthetic_dataset"]["losses"][
                dropout_rate
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for Dropout Rate: {dropout_rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_dropout_{dropout_rate}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()
