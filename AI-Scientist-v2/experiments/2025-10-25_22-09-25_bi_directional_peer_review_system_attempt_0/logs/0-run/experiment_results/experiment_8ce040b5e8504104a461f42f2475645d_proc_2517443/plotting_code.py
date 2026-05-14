import matplotlib.pyplot as plt
import numpy as np
import os

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Generate plots
for dataset_id in range(1, 4):
    try:
        dataset_key = f"dataset_{dataset_id}"
        losses = experiment_data["dataset_variability_impact"][dataset_key]["losses"][
            "train"
        ]
        metrics = experiment_data["dataset_variability_impact"][dataset_key]["metrics"][
            "train"
        ]

        # Plot losses
        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.title(f"Training Loss for {dataset_key}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_key}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_key}: {e}")
        plt.close()

    try:
        # Plot metrics (RQS)
        plt.figure()
        plt.plot(metrics, label="RQS Metric")
        plt.title(f"RQS Metric for {dataset_key}")
        plt.xlabel("Epochs")
        plt.ylabel("RQS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_key}_rqs_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RQS plot for {dataset_key}: {e}")
        plt.close()
