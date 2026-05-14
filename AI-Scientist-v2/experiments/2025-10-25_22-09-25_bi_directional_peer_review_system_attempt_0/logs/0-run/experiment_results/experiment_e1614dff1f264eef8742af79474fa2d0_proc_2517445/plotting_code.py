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

try:
    batch_sizes = [16, 32, 64, 128]
    for batch_size in batch_sizes:
        losses = experiment_data["impact_of_batch_size"]["synthetic_data"]["losses"][
            "train"
        ]
        plt.figure()
        plt.plot(range(len(losses)), losses, marker="o")
        plt.title(f"Training Loss for Batch Size {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(
            os.path.join(working_dir, f"training_loss_batch_size_{batch_size}.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")

try:
    metrics = experiment_data["impact_of_batch_size"]["synthetic_data"]["metrics"][
        "train"
    ]
    plt.figure()
    plt.plot(range(len(metrics)), metrics, marker="o", color="orange")
    plt.title("Training RQS Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.savefig(os.path.join(working_dir, "training_rqs_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQS metrics plot: {e}")
