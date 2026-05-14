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
    # Training Loss Plot
    plt.figure()
    epochs = list(
        range(
            1,
            len(
                experiment_data["hyperparam_tuning_hidden_layers"]["experiment"][
                    "losses"
                ]["train"]
            )
            + 1,
        )
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_hidden_layers"]["experiment"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_hidden_layers"]["experiment"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "experiment_training_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Validation Loss Plot
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_hidden_layers"]["experiment"]["losses"][
            "val"
        ],
        label="Validation Loss",
        color="orange",
    )
    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "experiment_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
