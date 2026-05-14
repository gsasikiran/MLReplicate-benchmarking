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

# Linear Dataset Loss Plot
try:
    linear_losses = experiment_data["multiple_datasets_evaluation"]["linear_dataset"][
        "losses"
    ]
    plt.figure()
    plt.plot(linear_losses["train"], label="Training Loss")
    plt.plot(linear_losses["val"], label="Validation Loss")
    plt.title("Loss Curves for Linear Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "linear_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating linear dataset loss plot: {e}")
    plt.close()

# Quadratic Dataset Loss Plot
try:
    quadratic_losses = experiment_data["multiple_datasets_evaluation"][
        "quadratic_dataset"
    ]["losses"]
    plt.figure()
    plt.plot(quadratic_losses["train"], label="Training Loss")
    plt.plot(quadratic_losses["val"], label="Validation Loss")
    plt.title("Loss Curves for Quadratic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "quadratic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating quadratic dataset loss plot: {e}")
    plt.close()

# Cubic Dataset Loss Plot
try:
    cubic_losses = experiment_data["multiple_datasets_evaluation"]["cubic_dataset"][
        "losses"
    ]
    plt.figure()
    plt.plot(cubic_losses["train"], label="Training Loss")
    plt.plot(cubic_losses["val"], label="Validation Loss")
    plt.title("Loss Curves for Cubic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "cubic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cubic dataset loss plot: {e}")
    plt.close()
