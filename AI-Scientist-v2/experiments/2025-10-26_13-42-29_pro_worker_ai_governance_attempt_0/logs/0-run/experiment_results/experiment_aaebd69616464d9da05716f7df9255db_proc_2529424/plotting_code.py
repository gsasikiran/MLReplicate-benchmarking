import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training loss
try:
    epochs = list(range(1, len(experiment_data["dataset_name"]["losses"]["train"]) + 1))
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["dataset_name"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dataset_name_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training metric
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["dataset_name"]["metrics"]["train"],
        label="Training Metric",
        color="orange",
    )
    plt.title("Training Metric Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dataset_name_training_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training metric plot: {e}")
    plt.close()
