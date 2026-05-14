import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    # Load experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training metrics
for noise_level in experiment_data["noise_robustness"]["baseline"]["losses"]["train"]:
    try:
        epochs = [5, 10, 20, 30]
        train_losses = experiment_data["noise_robustness"]["baseline"]["losses"][
            "train"
        ]
        train_accuracies = experiment_data["noise_robustness"]["baseline"]["metrics"][
            "train"
        ]

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, train_accuracies, label="Training Accuracy")
        plt.title(f"Noise Level: {noise_level} - Training Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"noise_{noise_level}_training_metrics.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for noise level {noise_level}: {e}")
        plt.close()
