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
    # Plot training and validation losses
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot training and validation metrics
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["metrics"][
            "train"
        ],
        label="Training Score",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["synthetic_data"]["metrics"]["val"],
        label="Validation Score",
    )
    plt.title("Training and Validation Scores")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_score_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating score plot: {e}")
    plt.close()
