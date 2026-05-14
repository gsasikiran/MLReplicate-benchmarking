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

reg_types = ["none", "l1", "l2", "dropout"]
for reg in reg_types:
    try:
        plt.figure()
        plt.plot(
            experiment_data[reg]["synthetic_dataset"]["losses"]["train"],
            label="Train Loss",
        )
        plt.title(f"{reg} Regularization - Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{reg}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {reg} training loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data[reg]["synthetic_dataset"]["metrics"]["train"],
            label="Train Accuracy",
        )
        plt.title(f"{reg} Regularization - Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{reg}_training_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {reg} training accuracy plot: {e}")
        plt.close()
