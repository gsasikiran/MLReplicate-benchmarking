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

for dr in experiment_data["dropout_ablation"]:
    dropout_data = experiment_data["dropout_ablation"][dr]

    # Plot training and validation losses
    try:
        plt.figure()
        plt.plot(dropout_data["losses"]["train"], label="Train Loss")
        plt.plot(dropout_data["losses"]["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {dr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_{dr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dr}: {e}")
        plt.close()

    # Scatter plot for ground truth vs predictions
    try:
        plt.figure()
        plt.scatter(dropout_data["ground_truth"], dropout_data["predictions"])
        plt.plot([0, 1], [0, 1], "r--")  # line y=x
        plt.title(f"Ground Truth vs Predictions for {dr}")
        plt.xlabel("Ground Truth WWBI")
        plt.ylabel("Predicted WWBI")
        plt.savefig(os.path.join(working_dir, f"ground_truth_vs_predictions_{dr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot for {dr}: {e}")
        plt.close()
