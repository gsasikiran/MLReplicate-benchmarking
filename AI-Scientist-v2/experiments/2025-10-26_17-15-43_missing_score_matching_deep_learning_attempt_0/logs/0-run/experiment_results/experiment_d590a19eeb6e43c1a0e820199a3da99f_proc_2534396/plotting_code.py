import matplotlib.pyplot as plt
import numpy as np
import os

# Define working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Iterate through different learning rates for plots
for lr in experiment_data["hyperparam_tuning_learning_rate"]:
    try:
        plt.figure()
        epochs = range(
            1,
            len(
                experiment_data["hyperparam_tuning_learning_rate"][lr]["losses"][
                    "train"
                ]
            )
            + 1,
        )
        plt.plot(
            epochs,
            experiment_data["hyperparam_tuning_learning_rate"][lr]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            epochs,
            experiment_data["hyperparam_tuning_learning_rate"][lr]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for Learning Rate {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for LR {lr}: {e}")
        plt.close()
