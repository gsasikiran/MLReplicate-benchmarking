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

try:
    epochs = range(
        1,
        len(
            experiment_data["batch_size_tuning"]["synthetic_dataset"]["losses"]["train"]
        )
        + 1,
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_dataset"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()
