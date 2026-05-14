import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training losses
try:
    losses = experiment_data["optimizer_variants"]["synthetic_dataset"]["losses"][
        "train"
    ]
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.title("Training Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training accuracies
try:
    metrics = experiment_data["optimizer_variants"]["synthetic_dataset"]["metrics"][
        "train"
    ]
    plt.figure()
    plt.plot(metrics, label="Training Accuracy", color="orange")
    plt.title("Training Accuracies Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracies.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
