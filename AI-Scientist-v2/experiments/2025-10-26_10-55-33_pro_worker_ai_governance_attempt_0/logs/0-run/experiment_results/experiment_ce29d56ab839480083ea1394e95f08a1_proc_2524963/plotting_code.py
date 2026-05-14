import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for hidden_units in experiment_data["varying_hidden_units"]:
    try:
        epochs = len(
            experiment_data["varying_hidden_units"][hidden_units]["losses"]["train"]
        )
        plt.figure()
        plt.plot(
            range(1, epochs + 1),
            experiment_data["varying_hidden_units"][hidden_units]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            range(1, epochs + 1),
            experiment_data["varying_hidden_units"][hidden_units]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves: Hidden Units = {hidden_units}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"Loss_Curves_HiddenUnits_{hidden_units}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot for hidden units {hidden_units}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            range(1, epochs + 1),
            experiment_data["varying_hidden_units"][hidden_units]["metrics"]["val"],
            label="PWIS",
        )
        plt.title(f"PWIS Metric: Hidden Units = {hidden_units}")
        plt.xlabel("Epoch")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"PWIS_HiddenUnits_{hidden_units}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PWIS plot for hidden units {hidden_units}: {e}")
        plt.close()
