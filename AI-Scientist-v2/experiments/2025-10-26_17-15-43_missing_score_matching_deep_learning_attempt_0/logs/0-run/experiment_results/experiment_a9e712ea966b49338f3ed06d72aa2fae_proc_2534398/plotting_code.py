import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses
try:
    plt.figure()
    epochs = range(
        1,
        len(
            experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()
