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

# Plot training loss
try:
    losses = experiment_data["impact_of_model_depth"]["synthetic_data"]["losses"][
        "train"
    ]
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training metrics (RQS)
try:
    metrics = experiment_data["impact_of_model_depth"]["synthetic_data"]["metrics"][
        "train"
    ]
    plt.figure()
    plt.plot(metrics, label="Training RQS", marker="o")
    plt.title("Training RQS per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training RQS plot: {e}")
    plt.close()
