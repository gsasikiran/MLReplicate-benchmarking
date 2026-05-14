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

# Plot Training Accuracy
try:
    plt.figure()
    accuracies = experiment_data["batch_size_variation"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    epochs = list(range(1, len(accuracies) + 1))
    plt.plot(epochs, accuracies, marker="o")
    plt.title("Training Accuracy per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(working_dir, "training_accuracy_synthetic_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()

# Plot Training Loss
try:
    plt.figure()
    losses = experiment_data["batch_size_variation"]["synthetic_dataset"]["losses"][
        "train"
    ]
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses, color="red", marker="x")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(working_dir, "training_loss_synthetic_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()
