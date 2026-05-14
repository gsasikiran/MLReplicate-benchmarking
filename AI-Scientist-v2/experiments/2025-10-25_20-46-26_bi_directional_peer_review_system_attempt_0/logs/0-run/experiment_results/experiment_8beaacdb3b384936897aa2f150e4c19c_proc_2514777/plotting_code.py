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
    # Plotting Training and Validation Losses
    train_losses = experiment_data["activation_function_tuning"]["FeedbackDataset"][
        "losses"
    ]["train"]
    val_losses = experiment_data["activation_function_tuning"]["FeedbackDataset"][
        "losses"
    ]["val"]

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.title("Loss Curves for Feedback Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plotting Training Metrics
    train_metrics = experiment_data["activation_function_tuning"]["FeedbackDataset"][
        "metrics"
    ]["train"]
    plt.figure()
    plt.plot(range(1, len(train_metrics) + 1), train_metrics, label="Training Metrics")
    plt.title("Training Metrics Over Epochs for Feedback Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_training_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
