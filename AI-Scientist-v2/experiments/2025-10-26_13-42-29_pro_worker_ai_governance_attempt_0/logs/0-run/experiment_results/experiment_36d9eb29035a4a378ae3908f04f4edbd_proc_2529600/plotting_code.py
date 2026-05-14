import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses
for batch_size in experiment_data.keys():
    try:
        plt.figure()
        plt.plot(experiment_data[batch_size]["losses"]["train"], label="Training Loss")
        plt.plot(experiment_data[batch_size]["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Batch Size: {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{batch_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {batch_size}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(experiment_data[batch_size]["metrics"]["val"], label="Validation WWBI")
        plt.title(f"Validation WWBI for Batch Size: {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("WWBI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"wwbi_curves_{batch_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating WWBI plot for {batch_size}: {e}")
        plt.close()
