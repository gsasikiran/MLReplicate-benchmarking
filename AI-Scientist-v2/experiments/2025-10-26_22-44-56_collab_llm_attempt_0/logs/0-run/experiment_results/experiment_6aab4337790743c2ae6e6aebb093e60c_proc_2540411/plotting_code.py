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

# Plot training losses
for dataset_name in experiment_data["multi_dataset_evaluation"].keys():
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][dataset_name]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.title(f"{dataset_name} - Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} losses: {e}")
        plt.close()

# Plot UES metrics
for dataset_name in experiment_data["multi_dataset_evaluation"].keys():
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][dataset_name]["metrics"][
                "train"
            ],
            label="UES Metric",
        )
        plt.title(f"{dataset_name} - UES Metric")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_ues_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} UES: {e}")
        plt.close()
