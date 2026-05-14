import matplotlib.pyplot as plt
import numpy as np
import os

# Working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Training Loss Plot
try:
    plt.figure()
    plt.plot(
        experiment_data["simple_nn_experiment"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["simple_nn_experiment"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_nn_training_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Training Metric Plot
try:
    plt.figure()
    plt.plot(
        experiment_data["simple_nn_experiment"]["metrics"]["train"],
        label="Training Metric",
    )
    plt.plot(
        experiment_data["simple_nn_experiment"]["metrics"]["val"],
        label="Validation Metric",
    )
    plt.title("Training and Validation Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_nn_training_validation_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training metrics plot: {e}")
    plt.close()
