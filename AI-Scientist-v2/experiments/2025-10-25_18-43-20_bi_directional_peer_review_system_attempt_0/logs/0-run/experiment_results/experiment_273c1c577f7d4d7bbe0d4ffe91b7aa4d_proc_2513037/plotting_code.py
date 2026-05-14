import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    plt.figure()
    epochs = range(
        len(
            experiment_data["hyperparam_tuning_num_hidden_units"]["synthetic_dataset"][
                "losses"
            ]["train"]
        )
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_num_hidden_units"]["synthetic_dataset"][
            "losses"
        ]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_num_hidden_units"]["synthetic_dataset"][
            "losses"
        ]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training and validation loss plot: {e}")
    plt.close()
