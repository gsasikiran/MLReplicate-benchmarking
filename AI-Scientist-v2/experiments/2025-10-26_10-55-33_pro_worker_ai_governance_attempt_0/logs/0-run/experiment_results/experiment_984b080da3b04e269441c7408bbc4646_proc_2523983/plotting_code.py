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
    activation_names = experiment_data["activation_function_tuning"].keys()
    for activation_name in activation_names:
        losses = experiment_data["activation_function_tuning"][activation_name][
            "losses"
        ]
        plt.figure()
        plt.plot(losses["train"], label="Train Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {activation_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{activation_name}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    for activation_name in activation_names:
        metrics = experiment_data["activation_function_tuning"][activation_name][
            "metrics"
        ]["val"]
        plt.figure()
        plt.plot(metrics, label="PWIS Metric")
        plt.title(f"Validation PWIS for {activation_name}")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"pwis_curves_{activation_name}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
