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

for noise_level in ["low_noise", "medium_noise", "high_noise"]:
    try:
        train_losses = experiment_data["multi_synthetic"][noise_level]["losses"][
            "train"
        ]
        val_losses = experiment_data["multi_synthetic"][noise_level]["losses"]["val"]
        epochs = list(range(1, len(train_losses) + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{noise_level} Dataset Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{noise_level}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {noise_level}: {e}")
        plt.close()
