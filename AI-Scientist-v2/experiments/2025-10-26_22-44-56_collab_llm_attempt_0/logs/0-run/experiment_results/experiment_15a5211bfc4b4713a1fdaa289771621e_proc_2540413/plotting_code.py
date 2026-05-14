import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training losses
for dataset_name in experiment_data["multi_dataset_generalization"]:
    try:
        losses = experiment_data["multi_dataset_generalization"][dataset_name][
            "losses"
        ]["train"]
        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.title(f"{dataset_name.capitalize()} Dataset Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} training losses: {e}")
        plt.close()

# Plot training metrics (e.g., UES)
for dataset_name in experiment_data["multi_dataset_generalization"]:
    try:
        metrics = experiment_data["multi_dataset_generalization"][dataset_name][
            "metrics"
        ]["train"]
        plt.figure()
        plt.plot(metrics, label="UES Metric")
        plt.title(f"{dataset_name.capitalize()} Dataset UES Metric")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_ues_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} UES metrics: {e}")
        plt.close()
