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
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["train"], label="Training Loss"
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training Loss plot: {e}")
    plt.close()

try:
    # CODS Metric Plot
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["metrics"]["train"], label="Train CODS"
    )
    plt.title("Train CODS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_train_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Train CODS plot: {e}")
    plt.close()
