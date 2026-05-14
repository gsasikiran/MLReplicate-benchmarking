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

try:
    # Plot training losses
    plt.figure()
    train_losses = experiment_data["hyperparam_tuning_activation_function"][
        "synthetic_dataset"
    ]["losses"]["train"]
    plt.plot(train_losses)
    plt.title("Training Losses for Synthetic Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training losses plot: {e}")
    plt.close()

try:
    # Plot training accuracies
    plt.figure()
    train_accuracies = experiment_data["hyperparam_tuning_activation_function"][
        "synthetic_dataset"
    ]["metrics"]["train"]
    plt.plot(train_accuracies)
    plt.title("Training Accuracies for Synthetic Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracies.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracies plot: {e}")
    plt.close()
