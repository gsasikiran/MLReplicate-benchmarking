import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for dataset, batch_data in experiment_data["multi_dataset_evaluation"].items():
    for batch_size, data in batch_data.items():
        try:
            # Plot training and validation losses
            plt.figure()
            plt.plot(data["losses"]["train"], label="Training Loss")
            plt.plot(data["losses"]["val"], label="Validation Loss")
            plt.title(f"{dataset} - Batch Size {batch_size}\nLosses over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_batch_{batch_size}_losses.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dataset}, batch {batch_size}: {e}")
            plt.close()

        try:
            # Plot validation PWIS metrics
            plt.figure()
            plt.plot(data["metrics"]["val"], label="Validation PWIS")
            plt.title(
                f"{dataset} - Batch Size {batch_size}\nValidation Metrics over Epochs"
            )
            plt.xlabel("Epochs")
            plt.ylabel("PWIS")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset}_batch_{batch_size}_pw_metrics.png"
                )
            )
            plt.close()
        except Exception as e:
            print(f"Error creating PWIS plot for {dataset}, batch {batch_size}: {e}")
            plt.close()
