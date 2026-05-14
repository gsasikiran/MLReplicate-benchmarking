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

# Plot Training Loss
try:
    plt.figure()
    for dropout_rate in experiment_data["dropout_variation"]["synthetic_dataset"][
        "losses"
    ]["train"]:
        plt.plot(dropout_rate, label=f"Dropout Rate {dropout_rate}")
    plt.title("Training Loss Across Dropout Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot Training Accuracy
try:
    plt.figure()
    for accuracy in experiment_data["dropout_variation"]["synthetic_dataset"][
        "metrics"
    ]["train"]:
        plt.plot(accuracy, label=f"Dropout Rate {dropout_rate}")
    plt.title("Training Accuracy Across Dropout Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
