import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for dataset_name, data in experiment_data["multi_dataset_evaluation"].items():
    try:
        epochs = list(range(1, len(data["losses"]["train"]) + 1))
        plt.figure()
        plt.plot(epochs, data["losses"]["train"], label="Training Loss", marker="o")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss", marker="x")
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {e}")
        plt.close()
