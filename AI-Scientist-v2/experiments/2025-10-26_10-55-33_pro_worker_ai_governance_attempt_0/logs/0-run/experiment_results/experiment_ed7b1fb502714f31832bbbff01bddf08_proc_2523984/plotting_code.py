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
    epochs = range(
        1,
        len(
            experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_dataset"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_dataset"][
            "losses"
        ]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_dataset"][
            "losses"
        ]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_training_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for training/validation loss: {e}")

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_dataset"][
            "metrics"
        ]["val"],
        label="Validation PWIS",
    )
    plt.title("Validation PWIS Metrics - Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_validation_pwis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for validation PWIS: {e}")
