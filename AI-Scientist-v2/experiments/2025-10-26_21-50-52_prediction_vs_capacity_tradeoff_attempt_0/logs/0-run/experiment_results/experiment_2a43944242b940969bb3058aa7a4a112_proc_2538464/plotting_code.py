import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training losses
try:
    plt.figure()
    plt.plot(
        experiment_data["layer_sizes_tuning"]["synthetic_dataset"]["losses"]["train"],
        label="Train Loss",
    )
    plt.title("Training Loss Curve")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot training accuracy
try:
    plt.figure()
    plt.plot(
        experiment_data["layer_sizes_tuning"]["synthetic_dataset"]["metrics"]["train"],
        label="Train Accuracy",
        color="g",
    )
    plt.title("Training Accuracy Curve")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_accuracy_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()
