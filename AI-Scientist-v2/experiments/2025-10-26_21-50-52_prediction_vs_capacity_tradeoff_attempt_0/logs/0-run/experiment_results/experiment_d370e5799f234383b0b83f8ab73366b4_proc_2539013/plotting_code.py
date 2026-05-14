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

# Plot training loss and accuracy for each dataset
for dataset_name, data in experiment_data["multiple_synthetic_datasets"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Train Loss")
        plt.title(f"Training Loss for {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_train_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} training loss: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="Train Accuracy")
        plt.title(f"Training Accuracy for {dataset_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_train_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} training accuracy: {e}")
        plt.close()

    # Plot sample predictions for a limited number of epochs
    if len(data["predictions"]) > 0:
        try:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.scatter(
                range(len(data["ground_truth"])),
                data["ground_truth"],
                color="blue",
                label="Ground Truth",
            )
            plt.title(f"Ground Truth for {dataset_name}")
            plt.subplot(1, 2, 2)
            plt.scatter(
                range(len(data["predictions"])),
                data["predictions"],
                color="orange",
                label="Predictions",
            )
            plt.title(f"Predictions for {dataset_name}")
            plt.suptitle(f"Left: Ground Truth, Right: Predictions for {dataset_name}")
            plt.savefig(os.path.join(working_dir, f"{dataset_name}_predictions.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating predictions plot for {dataset_name}: {e}")
            plt.close()
