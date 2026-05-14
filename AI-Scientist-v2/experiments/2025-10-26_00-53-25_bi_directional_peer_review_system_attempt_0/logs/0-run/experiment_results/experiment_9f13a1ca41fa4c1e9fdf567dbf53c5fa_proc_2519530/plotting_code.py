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
    epochs = range(1, len(experiment_data["synthetic_reviews"]["losses"]["train"]) + 1)
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_reviews"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_reviews"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_reviews_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()
