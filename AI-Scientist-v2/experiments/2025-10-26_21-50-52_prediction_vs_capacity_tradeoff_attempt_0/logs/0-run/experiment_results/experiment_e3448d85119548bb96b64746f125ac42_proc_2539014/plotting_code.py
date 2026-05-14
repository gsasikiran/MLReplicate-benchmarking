import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training loss and accuracy
for dist_name in experiment_data["data_distribution_impact"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["data_distribution_impact"][dist_name]["losses"]["train"],
            label="Training Loss",
        )
        plt.title(f"{dist_name.capitalize()} - Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dist_name}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dist_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["data_distribution_impact"][dist_name]["metrics"]["train"],
            label="Training Accuracy",
        )
        plt.title(f"{dist_name.capitalize()} - Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dist_name}_training_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dist_name}: {e}")
        plt.close()
