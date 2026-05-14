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

for activation, data in experiment_data["activation_functions"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for Activation: {activation}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{activation}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {activation}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(data["ground_truth"], data["predictions"][-1], alpha=0.5)
        plt.plot(
            [min(data["ground_truth"]), max(data["ground_truth"])],
            [min(data["ground_truth"]), max(data["ground_truth"])],
            "r--",
            lw=2,
        )  # Identity line
        plt.title(f"Predictions vs Ground Truth for Activation: {activation}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"predictions_vs_ground_truth_{activation}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {activation}: {e}")
        plt.close()
