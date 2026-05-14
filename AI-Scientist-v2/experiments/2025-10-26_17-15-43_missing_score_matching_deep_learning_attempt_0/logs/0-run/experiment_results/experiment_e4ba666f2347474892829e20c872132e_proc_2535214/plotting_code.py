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

for dropout_rate, metrics in experiment_data["dropout_regularization_effect"].items():
    try:
        plt.figure()
        plt.plot(metrics["losses"]["train"], label="Training Loss")
        plt.plot(metrics["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Dropout Rate: {dropout_rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"loss_curves_dropout_{dropout_rate}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for dropout {dropout_rate}: {e}")
        plt.close()
