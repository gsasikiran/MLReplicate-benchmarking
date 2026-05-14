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
    epochs = np.arange(
        1, len(experiment_data["synthetic_dataset"]["losses"]["train"]) + 1
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_dataset"]["metrics"]["val"],
        label="WWBI Metric",
    )
    plt.xlabel("Epochs")
    plt.ylabel("WWBI")
    plt.title("WWBI Metric Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_wwbi_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating WWBI metric plot: {e}")
    plt.close()
