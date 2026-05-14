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
    # Plot training losses
    epochs = len(
        experiment_data["embedding_variation"]["synthetic_dataset"]["losses"]["train"]
    )
    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        experiment_data["embedding_variation"]["synthetic_dataset"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training losses plot: {e}")

try:
    # Plot CODS metrics
    plt.figure()
    plt.plot(
        range(1, epochs + 1),
        experiment_data["embedding_variation"]["synthetic_dataset"]["metrics"]["train"],
        label="CODS",
    )
    plt.title("CODS Metrics over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CODS metrics plot: {e}")
