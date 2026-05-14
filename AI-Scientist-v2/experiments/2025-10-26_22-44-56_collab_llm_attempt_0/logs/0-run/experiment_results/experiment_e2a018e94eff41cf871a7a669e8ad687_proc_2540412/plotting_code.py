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

try:
    losses = experiment_data["input_feature_dimensionality_analysis"][
        "synthetic_dataset"
    ]["losses"]["train"]
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    metrics = experiment_data["input_feature_dimensionality_analysis"][
        "synthetic_dataset"
    ]["metrics"]["train"]
    plt.figure()
    plt.plot(metrics, label="Training UES Metric")
    plt.title("Training UES Metric Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("UES Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_ues.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training UES metric plot: {e}")
    plt.close()
