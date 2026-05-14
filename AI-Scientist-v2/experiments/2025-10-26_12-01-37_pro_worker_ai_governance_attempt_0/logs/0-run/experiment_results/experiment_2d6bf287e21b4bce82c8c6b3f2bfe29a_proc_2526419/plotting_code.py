import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses
try:
    for activation in experiment_data["activation_function_experiment"]:
        for hidden_units in experiment_data["activation_function_experiment"][
            activation
        ]:
            losses = experiment_data["activation_function_experiment"][activation][
                hidden_units
            ]["losses"]
            epochs = range(len(losses["train"]))

            plt.figure()
            plt.plot(epochs, losses["train"], label="Training Loss")
            plt.plot(epochs, losses["val"], label="Validation Loss")
            plt.title(f"{activation} Activation - Hidden Units: {hidden_units}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(working_dir, f"loss_plot_{activation}_{hidden_units}.png")
            )
            plt.close()
except Exception as e:
    print(f"Error creating loss plots: {e}")
    plt.close()

# Plot evaluation metrics
try:
    for activation in experiment_data["activation_function_experiment"]:
        for hidden_units in experiment_data["activation_function_experiment"][
            activation
        ]:
            metrics = experiment_data["activation_function_experiment"][activation][
                hidden_units
            ]["metrics"]
            epochs = range(len(metrics["val"]))

            plt.figure()
            plt.plot(epochs, metrics["val"], label="EIS Metric")
            plt.title(f"{activation} Activation - Hidden Units: {hidden_units}")
            plt.xlabel("Epochs")
            plt.ylabel("EIS Metric")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(
                    working_dir, f"metric_plot_{activation}_{hidden_units}.png"
                )
            )
            plt.close()
except Exception as e:
    print(f"Error creating metric plots: {e}")
    plt.close()
