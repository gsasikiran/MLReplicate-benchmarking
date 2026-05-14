import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for method in experiment_data["imputation_methods"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["imputation_methods"][method]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["imputation_methods"][method]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for {method.capitalize()} Imputation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{method}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {method}: {e}")
        plt.close()

for method in experiment_data["imputation_methods"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["imputation_methods"][method]["mdie"],
            label="MDIE",
            color="orange",
        )
        plt.title(f"MDIE for {method.capitalize()} Imputation")
        plt.xlabel("Epochs")
        plt.ylabel("MDIE")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"mdie_{method}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MDIE plot for {method}: {e}")
        plt.close()
