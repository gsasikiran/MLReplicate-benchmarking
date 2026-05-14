import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot loss curves
try:
    losses = experiment_data["dropout_ablation"]["FeedbackDataset"]["losses"]
    plt.figure()
    plt.plot(losses["train"], label="Training Loss")
    plt.plot(losses["val"], label="Validation Loss")
    plt.title("Loss Curves for Feedback Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_Loss_Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot metrics
try:
    metrics = experiment_data["dropout_ablation"]["FeedbackDataset"]["metrics"]
    plt.figure()
    plt.plot(metrics["train"], label="Training Metrics")
    plt.title("Training Metrics for Feedback Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_Training_Metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
