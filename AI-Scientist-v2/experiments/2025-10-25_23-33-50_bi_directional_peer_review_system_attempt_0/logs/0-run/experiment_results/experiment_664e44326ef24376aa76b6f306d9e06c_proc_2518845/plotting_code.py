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
for activation_name in experiment_data["activation_function_comparison"][
    "losses"
].keys():
    try:
        plt.figure()
        plt.plot(
            experiment_data["activation_function_comparison"]["losses"][
                activation_name
            ]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["activation_function_comparison"]["losses"][
                activation_name
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"{activation_name} Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation_name}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {activation_name} loss plot: {e}")
        plt.close()

# Plot RQI metrics
for activation_name in experiment_data["activation_function_comparison"][
    "metrics"
].keys():
    try:
        plt.figure()
        plt.plot(
            experiment_data["activation_function_comparison"]["metrics"][
                activation_name
            ]["train"],
            label="RQI (Train)",
        )
        plt.title(f"{activation_name} RQI Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("RQI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{activation_name}_rqi.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {activation_name} RQI plot: {e}")
        plt.close()
