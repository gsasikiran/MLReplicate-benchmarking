import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Training and validation loss plot
try:
    for dataset_name in experiment_data["multiple_synthetic_datasets"]:
        losses = experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"]
        epochs = range(len(losses["train"]))

        plt.figure()
        plt.plot(epochs, losses["train"], label="Training Loss")
        plt.plot(epochs, losses["val"], label="Validation Loss")
        plt.title(f"{dataset_name} - Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_plot.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Training and validation accuracy plot
try:
    for dataset_name in experiment_data["multiple_synthetic_datasets"]:
        metrics = experiment_data["multiple_synthetic_datasets"][dataset_name][
            "metrics"
        ]
        epochs = range(len(metrics["train"]))

        plt.figure()
        plt.plot(epochs, metrics["train"], label="Training Accuracy")
        plt.plot(epochs, metrics["val"], label="Validation Accuracy")
        plt.title(f"{dataset_name} - Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_accuracy_plot.png"))
        plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
