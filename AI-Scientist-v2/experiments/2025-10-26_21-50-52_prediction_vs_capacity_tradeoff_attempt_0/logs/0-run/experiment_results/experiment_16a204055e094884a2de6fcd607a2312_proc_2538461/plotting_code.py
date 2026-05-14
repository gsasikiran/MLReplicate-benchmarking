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

for rate in experiment_data.keys():
    try:
        plt.figure()
        plt.plot(experiment_data[rate]["losses"]["train"], label="Training Loss")
        plt.plot(experiment_data[rate]["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curve for Dropout Rate {rate}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_dropout_{rate}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {rate}: {e}")

    try:
        plt.figure()
        plt.plot(experiment_data[rate]["metrics"]["train"], label="Training Accuracy")
        plt.title(f"Accuracy Curve for Dropout Rate {rate}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"accuracy_curve_dropout_{rate}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {rate}: {e}")
