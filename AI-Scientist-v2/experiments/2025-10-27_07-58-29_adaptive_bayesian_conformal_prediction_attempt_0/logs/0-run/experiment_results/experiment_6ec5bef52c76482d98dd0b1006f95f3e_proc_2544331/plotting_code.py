import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plotting loss curves
for optimizer_name in experiment_data["optimizer_comparison"]:
    try:
        train_losses = experiment_data["optimizer_comparison"][optimizer_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["optimizer_comparison"][optimizer_name]["losses"][
            "val"
        ]

        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{optimizer_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{optimizer_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {optimizer_name} losses: {e}")
        plt.close()

    try:
        ground_truth = np.concatenate(
            experiment_data["optimizer_comparison"][optimizer_name]["ground_truth"]
        )
        predictions = np.concatenate(
            experiment_data["optimizer_comparison"][optimizer_name]["predictions"]
        )

        plt.figure()
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.title(f"{optimizer_name} Predictions vs Ground Truth")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted Values")
        plt.axis("equal")
        plt.savefig(
            os.path.join(
                working_dir, f"{optimizer_name}_predictions_vs_ground_truth.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {optimizer_name} predictions: {e}")
        plt.close()
