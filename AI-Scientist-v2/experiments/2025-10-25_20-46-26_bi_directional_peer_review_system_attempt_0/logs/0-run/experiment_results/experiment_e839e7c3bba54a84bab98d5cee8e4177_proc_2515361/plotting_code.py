import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training loss
try:
    train_losses = experiment_data["activation_function_variability"][
        "FeedbackDataset"
    ]["losses"]["train"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss over Epochs for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot validation loss
try:
    val_losses = experiment_data["activation_function_variability"]["FeedbackDataset"][
        "losses"
    ]["val"]
    plt.figure()
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Validation Loss over Epochs for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
