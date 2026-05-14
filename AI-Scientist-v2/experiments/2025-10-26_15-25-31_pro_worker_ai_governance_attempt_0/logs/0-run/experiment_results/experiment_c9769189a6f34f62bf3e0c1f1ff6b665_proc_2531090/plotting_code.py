import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    plt.plot(experiment_data["synthetic_data"]["losses"]["train"], label="Train Loss")
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")

try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["metrics"]["train"], label="Train Accuracy"
    )
    plt.plot(
        experiment_data["synthetic_data"]["metrics"]["val"], label="Validation Accuracy"
    )
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")

# More plots can be added depending on what data exists in experiment_data
