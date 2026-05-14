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

for init_method in experiment_data.keys():
    try:
        plt.figure()
        plt.plot(experiment_data[init_method]["losses"]["train"], label="Training Loss")
        plt.title(f"Training Loss over Epochs - {init_method}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{init_method}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {init_method} training loss: {e}")

    try:
        plt.figure()
        plt.plot(
            experiment_data[init_method]["metrics"]["train"], label="Training Accuracy"
        )
        plt.title(f"Training Accuracy over Epochs - {init_method}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{init_method}_training_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {init_method} training accuracy: {e}")
