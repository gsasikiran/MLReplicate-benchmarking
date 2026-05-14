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

# Plotting training and validation losses for each activation function
for act_name, batch_data in experiment_data["activation_functions"].items():
    for batch_size, data in batch_data.items():
        try:
            plt.figure()
            plt.plot(data["losses"]["train"], label="Train Loss")
            plt.plot(data["losses"]["val"], label="Validation Loss")
            plt.title(f"{act_name} Activation Function - Batch Size {batch_size}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir, f"{act_name}_batch_size_{batch_size}_losses.png"
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating loss plot for {act_name}, batch size {batch_size}: {e}"
            )
            plt.close()

# Plotting PWIS Metrics
for act_name, batch_data in experiment_data["activation_functions"].items():
    for batch_size, data in batch_data.items():
        try:
            plt.figure()
            plt.plot(data["metrics"]["val"], label="PWIS (Validation)")
            plt.title(f"{act_name} Activation Function - Batch Size {batch_size} PWIS")
            plt.xlabel("Epochs")
            plt.ylabel("PWIS")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir, f"{act_name}_batch_size_{batch_size}_PWIS.png"
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating PWIS plot for {act_name}, batch size {batch_size}: {e}"
            )
            plt.close()
