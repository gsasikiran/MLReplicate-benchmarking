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

for dataset_name in ["Uniform", "Normal", "Exponential"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name} Dataset Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_Loss_Curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    # Additional plots can be defined similarly...
