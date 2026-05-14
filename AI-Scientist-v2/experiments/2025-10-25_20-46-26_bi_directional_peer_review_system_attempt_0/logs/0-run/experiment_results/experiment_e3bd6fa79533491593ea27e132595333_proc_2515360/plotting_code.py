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

for method in experiment_data["synthetic_dataset_variability"]:
    plt.figure()
    epochs = range(
        1,
        len(experiment_data["synthetic_dataset_variability"][method]["losses"]["train"])
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset_variability"][method]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset_variability"][method]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title(f"Loss Curves for {method} Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"Loss_Curves_{method}.png"))
    plt.close()

for method in experiment_data["synthetic_dataset_variability"]:
    for act_name in experiment_data["synthetic_dataset_variability"][method]["metrics"]:
        try:
            plt.figure()
            plt.plot(
                experiment_data["synthetic_dataset_variability"][method]["metrics"][
                    act_name
                ],
                marker="o",
            )
            plt.title(f"Metrics for {method} Dataset with {act_name} Activation")
            plt.xlabel("Epochs")
            plt.ylabel("Metric Value")
            plt.savefig(os.path.join(working_dir, f"Metrics_{method}_{act_name}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating metrics plot for {method} with {act_name}: {e}")
            plt.close()
