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

for outlier_fraction, data in experiment_data["Outlier Impact Assessment"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.title(f"Training Loss for {outlier_fraction}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{outlier_fraction}_train_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for training loss: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="Training Accuracy")
        plt.title(f"Training Accuracy for {outlier_fraction}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{outlier_fraction}_train_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for training accuracy: {e}")
        plt.close()
