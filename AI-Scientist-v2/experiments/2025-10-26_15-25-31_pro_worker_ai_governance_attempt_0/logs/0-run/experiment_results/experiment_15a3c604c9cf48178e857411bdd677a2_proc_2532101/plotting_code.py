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
try:
    epochs = range(
        len(
            experiment_data["activation_function_ablation"]["synthetic_data"]["losses"][
                "train"
            ]
        )
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["activation_function_ablation"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["activation_function_ablation"]["synthetic_data"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot training and validation metrics
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["activation_function_ablation"]["synthetic_data"]["metrics"][
            "train"
        ],
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        experiment_data["activation_function_ablation"]["synthetic_data"]["metrics"][
            "val"
        ],
        label="Validation Accuracy",
    )
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
