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

for act_name in experiment_data["activation_function_tuning"].keys():
    try:
        plt.figure()
        plt.plot(
            experiment_data["activation_function_tuning"][act_name]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["activation_function_tuning"][act_name]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{act_name} Activation Function Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name}: {e}")
        plt.close()
