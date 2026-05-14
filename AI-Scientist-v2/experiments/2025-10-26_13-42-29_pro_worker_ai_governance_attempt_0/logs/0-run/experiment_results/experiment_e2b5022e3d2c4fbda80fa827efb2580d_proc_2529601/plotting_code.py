import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Loop through hidden layer sizes for loss and metrics plotting
for hidden_layer_size in experiment_data["hidden_layer_size_tuning"]:
    losses = experiment_data["hidden_layer_size_tuning"][hidden_layer_size]["losses"]
    metrics = experiment_data["hidden_layer_size_tuning"][hidden_layer_size]["metrics"]

    try:
        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Hidden Layer Size {hidden_layer_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"loss_curves_hidden_layer_size_{hidden_layer_size}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for losses: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(metrics["val"], label="Validation WWBI")
        plt.title(f"WWBI Metric for Hidden Layer Size {hidden_layer_size}")
        plt.xlabel("Epochs")
        plt.ylabel("WWBI")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"wwbi_metric_hidden_layer_size_{hidden_layer_size}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for WWBI: {e}")
        plt.close()
