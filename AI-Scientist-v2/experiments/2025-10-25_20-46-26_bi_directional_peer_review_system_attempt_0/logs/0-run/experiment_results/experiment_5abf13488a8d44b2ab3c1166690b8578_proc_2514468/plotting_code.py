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
        1,
        len(experiment_data["weight_initialization"]["xavier"]["losses"]["train"]) + 1,
    )
    plt.plot(
        epochs,
        experiment_data["weight_initialization"]["xavier"]["losses"]["train"],
        label="Xavier Train",
    )
    plt.plot(
        epochs,
        experiment_data["weight_initialization"]["xavier"]["losses"]["val"],
        label="Xavier Validation",
    )
    plt.title("Xavier Initialization - Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "xavier_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for Xavier: {e}")
    plt.close()

try:
    plt.figure()
    epochs = range(
        1, len(experiment_data["weight_initialization"]["he"]["losses"]["train"]) + 1
    )
    plt.plot(
        epochs,
        experiment_data["weight_initialization"]["he"]["losses"]["train"],
        label="He Train",
    )
    plt.plot(
        epochs,
        experiment_data["weight_initialization"]["he"]["losses"]["val"],
        label="He Validation",
    )
    plt.title("He Initialization - Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "he_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for He: {e}")
    plt.close()
