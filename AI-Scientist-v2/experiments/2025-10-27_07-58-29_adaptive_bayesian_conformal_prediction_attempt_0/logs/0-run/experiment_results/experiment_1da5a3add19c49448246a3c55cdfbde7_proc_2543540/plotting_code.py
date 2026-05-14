import matplotlib.pyplot as plt
import numpy as np
import os

# Working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses
try:
    epochs = range(
        1,
        len(
            experiment_data["hyperparam_tuning_weight_decay"]["synthetic_data"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_weight_decay"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_weight_decay"]["synthetic_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Training vs Validation Loss for Synthetic Data")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot reliability measure
try:
    reliability_measures = experiment_data["hyperparam_tuning_weight_decay"][
        "synthetic_data"
    ]["metrics"]["val"]
    plt.figure()
    plt.bar(range(len(reliability_measures)), reliability_measures)
    plt.title("Validation Reliability Measure for Synthetic Data")
    plt.xlabel("Weight Decay Variants")
    plt.ylabel("Reliability Measure")
    plt.xticks(range(len(reliability_measures)), ["0.0", "0.01", "0.1", "1.0"])
    plt.savefig(os.path.join(working_dir, "synthetic_data_reliability_measure.png"))
    plt.close()
except Exception as e:
    print(f"Error creating reliability measure plot: {e}")
    plt.close()
