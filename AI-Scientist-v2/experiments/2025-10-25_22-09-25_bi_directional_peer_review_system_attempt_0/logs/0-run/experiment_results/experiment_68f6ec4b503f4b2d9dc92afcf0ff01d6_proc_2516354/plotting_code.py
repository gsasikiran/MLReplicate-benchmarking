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

# Plot Training Loss
try:
    plt.figure()
    weight_decay_values = [0.0, 0.01, 0.1, 0.5]
    for wd in weight_decay_values:
        losses = experiment_data["weight_decay_tuning"]["synthetic_data"]["losses"][
            "train"
        ]
        plt.plot(range(1, len(losses) + 1), losses, label=f"wd={wd}")
    plt.title("Training Loss per Weight Decay")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot Training Metrics (RQS)
try:
    plt.figure()
    for wd in weight_decay_values:
        metrics = experiment_data["weight_decay_tuning"]["synthetic_data"]["metrics"][
            "train"
        ]
        plt.plot(range(1, len(metrics) + 1), metrics, label=f"wd={wd}")
    plt.title("Training RQS per Weight Decay")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_rqs_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training RQS plot: {e}")
    plt.close()
