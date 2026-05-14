import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for num_layers in range(1, 6):
    try:
        training_losses = experiment_data[f"{num_layers}_layers"]["losses"]["train"]
        validation_losses = experiment_data[f"{num_layers}_layers"]["losses"]["val"]

        plt.figure()
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.title(f"Loss Curves for {num_layers} Layers")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{num_layers}_layers.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {num_layers} layers: {e}")
        plt.close()
