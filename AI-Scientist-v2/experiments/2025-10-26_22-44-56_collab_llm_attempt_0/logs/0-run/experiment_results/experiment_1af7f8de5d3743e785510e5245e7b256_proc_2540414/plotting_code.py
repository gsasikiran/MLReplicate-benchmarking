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

for dataset_type in experiment_data["Dataset_Diversity_Analysis"]:
    losses = experiment_data["Dataset_Diversity_Analysis"][dataset_type]["losses"][
        "train"
    ]

    try:
        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.title(f"Training Loss for {dataset_type}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_type}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_type}: {e}")
        plt.close()

    try:
        # If UES metrics are available for plotting
        metrics = experiment_data["Dataset_Diversity_Analysis"][dataset_type][
            "metrics"
        ]["train"]
        plt.figure()
        plt.plot(metrics, label="UES Metric", color="orange")
        plt.title(f"UES Metric for {dataset_type}")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_type}_ues_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating UES plot for {dataset_type}: {e}")
        plt.close()
