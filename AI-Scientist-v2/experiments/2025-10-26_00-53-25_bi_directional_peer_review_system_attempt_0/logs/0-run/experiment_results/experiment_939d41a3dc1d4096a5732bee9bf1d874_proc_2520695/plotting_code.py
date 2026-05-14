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

# Plot training and validation metrics for original data
try:
    metrics_train = experiment_data["input_dimensionality_reduction"]["original_data"][
        "metrics"
    ]["train"]
    metrics_val = experiment_data["input_dimensionality_reduction"]["original_data"][
        "metrics"
    ]["val"]
    epochs = range(1, len(metrics_train) + 1)

    plt.figure()
    plt.plot(epochs, metrics_train, label="Training Accuracy")
    plt.plot(epochs, metrics_val, label="Validation Accuracy")
    plt.title("Original Data - Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "original_data_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot for original data: {e}")
    plt.close()

# Plot training and validation losses for original data
try:
    losses_train = experiment_data["input_dimensionality_reduction"]["original_data"][
        "losses"
    ]["train"]
    losses_val = experiment_data["input_dimensionality_reduction"]["original_data"][
        "losses"
    ]["val"]

    plt.figure()
    plt.plot(epochs, losses_train, label="Training Loss")
    plt.plot(epochs, losses_val, label="Validation Loss")
    plt.title("Original Data - Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "original_data_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot for original data: {e}")
    plt.close()

# Plot training and validation metrics for reduced data
try:
    metrics_train_reduced = experiment_data["input_dimensionality_reduction"][
        "reduced_data"
    ]["metrics"]["train"]
    metrics_val_reduced = experiment_data["input_dimensionality_reduction"][
        "reduced_data"
    ]["metrics"]["val"]

    plt.figure()
    plt.plot(epochs, metrics_train_reduced, label="Training Accuracy")
    plt.plot(epochs, metrics_val_reduced, label="Validation Accuracy")
    plt.title("Reduced Data - Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "reduced_data_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot for reduced data: {e}")
    plt.close()

# Plot training and validation losses for reduced data
try:
    losses_train_reduced = experiment_data["input_dimensionality_reduction"][
        "reduced_data"
    ]["losses"]["train"]
    losses_val_reduced = experiment_data["input_dimensionality_reduction"][
        "reduced_data"
    ]["losses"]["val"]

    plt.figure()
    plt.plot(epochs, losses_train_reduced, label="Training Loss")
    plt.plot(epochs, losses_val_reduced, label="Validation Loss")
    plt.title("Reduced Data - Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "reduced_data_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot for reduced data: {e}")
    plt.close()
