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

# Plot training and validation losses
for activation_name in experiment_data["activation_function_tuning"]["synthetic_data"][
    "losses"
].keys():
    try:
        losses = experiment_data["activation_function_tuning"]["synthetic_data"][
            "losses"
        ][activation_name]
        epochs = range(len(losses["train"]))

        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Validation Loss")
        plt.title(f"{activation_name.capitalize()} Activation Function Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation_name}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {activation_name} losses: {e}")

# Plot training and validation accuracy
for activation_name in experiment_data["activation_function_tuning"]["synthetic_data"][
    "metrics"
].keys():
    try:
        metrics = experiment_data["activation_function_tuning"]["synthetic_data"][
            "metrics"
        ][activation_name]
        epochs = range(len(metrics["train"]))

        plt.figure()
        plt.plot(epochs, metrics["train"], label="Train Accuracy")
        plt.plot(epochs, metrics["val"], label="Validation Accuracy")
        plt.title(f"{activation_name.capitalize()} Activation Function Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation_name}_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {activation_name} accuracy: {e}")
