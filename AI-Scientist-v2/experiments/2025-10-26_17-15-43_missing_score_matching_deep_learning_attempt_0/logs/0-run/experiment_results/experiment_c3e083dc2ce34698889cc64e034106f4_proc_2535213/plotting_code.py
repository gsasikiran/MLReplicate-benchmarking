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

for activation_name, sizes in experiment_data["Effect of Activation Functions"].items():
    for hidden_layer_size, data in sizes.items():
        try:
            plt.figure()
            epochs = range(1, len(data["losses"]["train"]) + 1)
            plt.plot(epochs, data["losses"]["train"], label="Training Loss")
            plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
            plt.title(
                f"Loss Curves for Activation: {activation_name}, Hidden Size: {hidden_layer_size}"
            )
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"Loss_Curve_Activation_{activation_name}_Size_{hidden_layer_size}.png",
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating loss plot for {activation_name}, size {hidden_layer_size}: {e}"
            )
            plt.close()
