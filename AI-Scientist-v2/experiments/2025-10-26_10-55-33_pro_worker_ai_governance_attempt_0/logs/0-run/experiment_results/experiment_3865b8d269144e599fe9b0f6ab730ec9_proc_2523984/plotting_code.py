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

weight_decay_values = list(experiment_data["hyperparam_tuning_weight_decay"].keys())
data = experiment_data["hyperparam_tuning_weight_decay"]["synthetic_dataset"]

try:
    plt.figure()
    for weight_decay in weight_decay_values:
        plt.plot(data["losses"]["train"], label=f"Train (wd={weight_decay})")
        plt.plot(data["losses"]["val"], label=f"Validation (wd={weight_decay})")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dataset_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(data["metrics"]["val"], label="PWIS")
    plt.title("Validation PWIS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_validation_pw.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
