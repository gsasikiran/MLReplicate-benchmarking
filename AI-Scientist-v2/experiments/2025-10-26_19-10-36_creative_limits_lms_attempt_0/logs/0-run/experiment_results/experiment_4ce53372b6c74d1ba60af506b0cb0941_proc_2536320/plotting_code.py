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

try:
    # Training Loss Plot
    training_losses = experiment_data["gradient_accumulation_tuning"][
        "synthetic_dataset"
    ]["losses"]["train"]
    plt.figure()
    plt.plot(training_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # CODS Plot
    training_metrics = experiment_data["gradient_accumulation_tuning"][
        "synthetic_dataset"
    ]["metrics"]["train"]
    plt.figure()
    plt.plot(training_metrics, label="CODS")
    plt.title("CODS Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CODS plot: {e}")
    plt.close()
