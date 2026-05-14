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

activation_names = list(experiment_data["activation_function_variation"].keys())

for act_name in activation_names:
    try:
        plt.figure()
        plt.plot(
            experiment_data["activation_function_variation"][act_name]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["activation_function_variation"][act_name]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for Activation Function: {act_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name}: {e}")
        plt.close()  # Always close figure even if error occurs
