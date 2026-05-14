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

if "hyperparam_tuning_hidden_layer_size" in experiment_data:
    dataset_results = experiment_data["hyperparam_tuning_hidden_layer_size"][
        "synthetic_dataset"
    ]

    try:
        plt.figure()
        plt.plot(dataset_results["losses"]["train"], label="Training Loss")
        plt.title("Training Losses for Synthetic Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "Synthetic_Dataset_Training_Loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            dataset_results["metrics"]["train"], label="Training UES", color="orange"
        )
        plt.title("Training UES for Synthetic Dataset")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "Synthetic_Dataset_Training_UES.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating UES plot: {e}")
        plt.close()
