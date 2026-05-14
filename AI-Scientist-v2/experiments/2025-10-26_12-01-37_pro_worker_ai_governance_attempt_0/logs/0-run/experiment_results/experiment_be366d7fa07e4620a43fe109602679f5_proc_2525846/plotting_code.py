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
    plt.figure()
    epochs = range(
        len(
            experiment_data["weight_decay_tuning"]["synthetic_dataset"]["losses"][
                "train"
            ]
        )
    )
    plt.plot(
        epochs,
        experiment_data["weight_decay_tuning"]["synthetic_dataset"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["weight_decay_tuning"]["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["weight_decay_tuning"]["synthetic_dataset"]["metrics"]["val"],
        label="Validation EIS",
    )
    plt.title("Validation Economic Impact Score (EIS) - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("EIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_validation_eis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation EIS plot: {e}")
    plt.close()
