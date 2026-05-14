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
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["sudoku"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_type_1"]["sudoku"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves for Sudoku Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Assuming additional metrics exist, plot them similarly
    # (Metrics extraction is hypothetical as actual metrics data is not outlined)
    metrics_train = experiment_data["hyperparam_tuning_type_1"]["sudoku"]["metrics"][
        "train"
    ]
    metrics_val = experiment_data["hyperparam_tuning_type_1"]["sudoku"]["metrics"][
        "val"
    ]

    plt.figure()
    plt.plot(metrics_train, label="Training Metrics")
    plt.plot(metrics_val, label="Validation Metrics")
    plt.title("Metrics Curves for Sudoku Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Metrics Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_metrics_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
