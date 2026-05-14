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
for feature in range(4):
    try:
        plt.figure()
        losses = experiment_data["feature_importance_removal"]["FeedbackDataset"][
            "losses"
        ]["train"]
        plt.plot(losses, label=f"Feature Removed: {feature}")
        plt.title("Training Loss Over Epochs - FeedbackDataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_feature_{feature}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot for feature {feature}: {e}")
        plt.close()

# Plot Validation Loss
for feature in range(4):
    try:
        plt.figure()
        val_losses = experiment_data["feature_importance_removal"]["FeedbackDataset"][
            "losses"
        ]["val"]
        plt.plot(val_losses, label=f"Feature Removed: {feature}")
        plt.title("Validation Loss Over Epochs - FeedbackDataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"validation_loss_feature_{feature}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot for feature {feature}: {e}")
        plt.close()
