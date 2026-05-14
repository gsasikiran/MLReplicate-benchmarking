import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for act_name in experiment_data["learning_rate_sensitivity"]:
    try:
        losses = experiment_data["learning_rate_sensitivity"][act_name]["losses"]
        epochs = np.arange(len(losses["train"]))

        plt.figure()
        plt.plot(epochs, losses["train"], label="Training Loss")
        plt.plot(epochs, losses["val"], label="Validation Loss")
        plt.title(f"{act_name} Activation Function")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name}: {e}")
        plt.close()
