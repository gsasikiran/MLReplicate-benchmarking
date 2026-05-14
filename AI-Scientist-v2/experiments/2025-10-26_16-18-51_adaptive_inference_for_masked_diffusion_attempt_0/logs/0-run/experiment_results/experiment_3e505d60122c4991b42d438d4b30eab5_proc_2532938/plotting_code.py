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

# Plot training and validation losses
try:
    plt.figure()
    epochs = range(
        1,
        len(
            experiment_data["hyperparam_tuning_num_epochs"]["sudoku"]["losses"]["train"]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_num_epochs"]["sudoku"]["losses"]["train"],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["hyperparam_tuning_num_epochs"]["sudoku"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Losses Over Epochs - Sudoku Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Further plotting functionality can be added as required for other data types
