import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    plt.plot(
        experiment_data["baseline_model"]["sudoku"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["baseline_model"]["sudoku"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Baseline Model Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "baseline_model_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating baseline model plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["additional_layers_model"]["sudoku"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["additional_layers_model"]["sudoku"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Additional Layers Model Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "additional_layers_model_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating additional layers model plot: {e}")
    plt.close()
