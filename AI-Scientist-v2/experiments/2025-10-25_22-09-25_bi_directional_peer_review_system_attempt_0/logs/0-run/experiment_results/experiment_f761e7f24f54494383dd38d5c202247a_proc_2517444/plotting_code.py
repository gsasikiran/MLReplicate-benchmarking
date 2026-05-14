import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    # Load experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Plot training losses
    plt.figure()
    plt.plot(
        experiment_data["learning_rate_variations"]["synthetic_data"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Plot training metrics (RQS)
    plt.figure()
    plt.plot(
        experiment_data["learning_rate_variations"]["synthetic_data"]["metrics"][
            "train"
        ],
        label="Training RQS",
    )
    plt.title("Training RQS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training RQS plot: {e}")
    plt.close()

try:
    # Plot predictions vs ground truth
    predictions = np.concatenate(
        experiment_data["learning_rate_variations"]["synthetic_data"]["predictions"]
    )
    ground_truth = np.concatenate(
        experiment_data["learning_rate_variations"]["synthetic_data"]["ground_truth"]
    )

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
