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

for hidden_layer_size in experiment_data["hyperparam_tuning_hidden_layer_size"].keys():
    try:
        losses = experiment_data["hyperparam_tuning_hidden_layer_size"][
            hidden_layer_size
        ]["losses"]
        epochs = list(range(1, len(losses["train"]) + 1))

        plt.figure()
        plt.plot(epochs, losses["train"], label="Training Loss")
        plt.plot(epochs, losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Hidden Layer Size: {hidden_layer_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"Loss_Curve_{hidden_layer_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {hidden_layer_size}: {e}")
        plt.close()
