import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training loss
try:
    plt.figure()
    weight_decays = experiment_data["weight_decay_tuning"]["synthetic_dataset"][
        "losses"
    ]["train"]
    for wd, losses in zip([0.0, 0.001, 0.01, 0.1], weight_decays):
        plt.plot(range(1, len(losses) + 1), losses, label=f"Weight Decay: {wd}")
    plt.title("Training Loss per Weight Decay")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plotting training accuracy
try:
    plt.figure()
    training_accuracy = experiment_data["weight_decay_tuning"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    for wd, accuracies in zip([0.0, 0.001, 0.01, 0.1], training_accuracy):
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f"Weight Decay: {wd}")
    plt.title("Training Accuracy per Weight Decay")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
