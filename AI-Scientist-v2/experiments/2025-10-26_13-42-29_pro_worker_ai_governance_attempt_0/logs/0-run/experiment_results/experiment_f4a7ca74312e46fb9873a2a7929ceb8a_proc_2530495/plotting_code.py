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

# Plot training and validation losses
try:
    plt.figure()
    for feature, data in experiment_data["feature_importance_ablation"].items():
        plt.plot(data["losses"]["train"], label=f"Train Loss - {feature}")
        plt.plot(
            data["losses"]["val"], label=f"Validation Loss - {feature}", linestyle="--"
        )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "losses_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating losses plot: {e}")
    plt.close()

# Plot WWBI metrics
try:
    plt.figure()
    for feature, data in experiment_data["feature_importance_ablation"].items():
        plt.plot(data["metrics"]["val"], label=f"WWBI - {feature}")
    plt.title("WWBI over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("WWBI Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "wwbi_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating WWBI plot: {e}")
    plt.close()
