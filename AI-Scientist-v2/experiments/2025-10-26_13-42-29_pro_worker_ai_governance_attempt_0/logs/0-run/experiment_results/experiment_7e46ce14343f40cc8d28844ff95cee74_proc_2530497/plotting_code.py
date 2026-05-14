import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Training and validation loss plots
for model_name in ["shallow", "moderate", "deep"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["neural_network_ablation"][model_name]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["neural_network_ablation"][model_name]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{model_name.capitalize()} Model Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {model_name}: {e}")
        plt.close()

# Predictions vs Ground Truth plots
for model_name in ["shallow", "moderate", "deep"]:
    try:
        plt.figure()
        plt.scatter(
            experiment_data["neural_network_ablation"][model_name]["ground_truth"],
            experiment_data["neural_network_ablation"][model_name]["predictions"],
            alpha=0.7,
        )
        plt.plot([0, 1], [0, 1], "r--")  # Line representing ideal prediction
        plt.title(f"{model_name.capitalize()} Model Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.axis("equal")
        plt.savefig(
            os.path.join(working_dir, f"{model_name}_predictions_vs_ground_truth.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {model_name}: {e}")
        plt.close()
