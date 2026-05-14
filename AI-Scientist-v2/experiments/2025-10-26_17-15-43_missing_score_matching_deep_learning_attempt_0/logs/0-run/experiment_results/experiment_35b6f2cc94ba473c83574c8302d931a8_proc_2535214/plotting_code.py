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

for pattern in ["mcar", "mar", "nmar"]:
    try:
        plt.figure()
        train_losses = experiment_data["missing_data_patterns"][pattern]["losses"][
            "train"
        ]
        val_losses = experiment_data["missing_data_patterns"][pattern]["losses"]["val"]
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{pattern.capitalize()} Dataset Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{pattern}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {pattern} loss plot: {e}")
        plt.close()
