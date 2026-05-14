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
    # Plot Training and Validation Loss
    plt.figure()
    epochs = range(
        len(experiment_data["batch_size_tuning"]["synthetic_data"]["losses"]["train"])
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_data"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Plot Training and Validation Metrics
    plt.figure()
    epochs = range(
        len(experiment_data["batch_size_tuning"]["synthetic_data"]["metrics"]["train"])
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_data"]["metrics"]["train"],
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        experiment_data["batch_size_tuning"]["synthetic_data"]["metrics"]["val"],
        label="Validation Accuracy",
    )
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_accuracy_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
