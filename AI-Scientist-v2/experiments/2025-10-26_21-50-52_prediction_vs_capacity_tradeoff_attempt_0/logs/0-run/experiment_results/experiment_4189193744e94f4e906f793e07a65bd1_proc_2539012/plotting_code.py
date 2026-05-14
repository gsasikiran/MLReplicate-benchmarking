import matplotlib.pyplot as plt
import numpy as np
import os

# Working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training losses
try:
    losses = experiment_data["activation_function_variation"]["synthetic_dataset"][
        "losses"
    ]["train"]
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training accuracy
try:
    accuracy = experiment_data["activation_function_variation"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    plt.figure()
    plt.plot(accuracy, label="Training Accuracy", color="orange")
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
