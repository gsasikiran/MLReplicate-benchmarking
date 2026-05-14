import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

hidden_layer_sizes = experiment_data["hyperparam_tuning"]["hidden_layer_size"].keys()

# Plot loss curves
for size in hidden_layer_sizes:
    try:
        losses = experiment_data["hyperparam_tuning"]["hidden_layer_size"][size][
            "losses"
        ]
        epochs = range(1, len(losses["train"]) + 1)
        plt.figure()
        plt.plot(epochs, losses["train"], label="Training Loss")
        plt.plot(epochs, losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Hidden Layer Size {size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"hidden_layer_{size}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for size {size}: {e}")
        plt.close()

# Plot predictions vs ground truth for the last hidden layer size
size = max(hidden_layer_sizes)  # Take the last size for illustration
try:
    predictions = experiment_data["hyperparam_tuning"]["hidden_layer_size"][size][
        "predictions"
    ]
    ground_truth = experiment_data["hyperparam_tuning"]["hidden_layer_size"][size][
        "ground_truth"
    ]
    plt.figure()
    plt.scatter(ground_truth, predictions)
    plt.plot(
        [min(ground_truth), max(ground_truth)],
        [min(ground_truth), max(ground_truth)],
        color="red",
        linestyle="--",
    )
    plt.title(f"Predictions vs Ground Truth for Hidden Layer Size {size}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(
            working_dir, f"hidden_layer_{size}_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot for size {size}: {e}")
    plt.close()
