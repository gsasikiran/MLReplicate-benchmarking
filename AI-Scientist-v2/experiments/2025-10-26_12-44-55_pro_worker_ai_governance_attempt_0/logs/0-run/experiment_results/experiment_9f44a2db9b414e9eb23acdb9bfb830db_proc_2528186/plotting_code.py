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
    plt.plot(experiment_data["unscaled"]["losses"]["train"], label="Train Loss")
    plt.plot(experiment_data["unscaled"]["losses"]["val"], label="Validation Loss")
    plt.title("Unscaled Data Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "unscaled_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(experiment_data["standardized"]["losses"]["train"], label="Train Loss")
    plt.plot(experiment_data["standardized"]["losses"]["val"], label="Validation Loss")
    plt.title("Standardized Data Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "standardized_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(experiment_data["normalized"]["losses"]["train"], label="Train Loss")
    plt.plot(experiment_data["normalized"]["losses"]["val"], label="Validation Loss")
    plt.title("Normalized Data Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "normalized_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

try:
    epochs = np.arange(len(experiment_data["unscaled"]["metrics"]["val"]))
    plt.figure()
    plt.plot(
        epochs, experiment_data["unscaled"]["metrics"]["val"], label="PWIS (Unscaled)"
    )
    plt.title("Pro-Worker Impact Score for Unscaled Data")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "unscaled_pwis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["standardized"]["metrics"]["val"],
        label="PWIS (Standardized)",
    )
    plt.title("Pro-Worker Impact Score for Standardized Data")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "standardized_pwis.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()
