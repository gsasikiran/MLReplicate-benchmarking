import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot Losses for each regularization method
for regularization_type in experiment_data["regularization"]:
    try:
        plt.figure()
        epochs = np.arange(
            len(
                experiment_data["regularization"][regularization_type]["losses"][
                    "train"
                ]
            )
        )
        plt.plot(
            epochs,
            experiment_data["regularization"][regularization_type]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            epochs,
            experiment_data["regularization"][regularization_type]["losses"]["val"],
            label="Val Loss",
        )
        plt.title(f"{regularization_type.capitalize()} Regularization Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{regularization_type}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {regularization_type}: {e}")
        plt.close()

# Plot Metrics for each regularization method
for regularization_type in experiment_data["regularization"]:
    try:
        plt.figure()
        epochs = np.arange(
            len(
                experiment_data["regularization"][regularization_type]["metrics"][
                    "train"
                ]
            )
        )
        plt.plot(
            epochs,
            experiment_data["regularization"][regularization_type]["metrics"]["train"],
            label="Train Metric",
        )
        plt.plot(
            epochs,
            experiment_data["regularization"][regularization_type]["metrics"]["val"],
            label="Val Metric",
        )
        plt.title(f"{regularization_type.capitalize()} Regularization Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{regularization_type}_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {regularization_type}: {e}")
        plt.close()
