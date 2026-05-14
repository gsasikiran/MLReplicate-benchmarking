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

for num_features in experiment_data["input_feature_variation"]:
    feature_str = num_features.replace("_features", "")

    # Plot training loss
    try:
        epochs = [5, 10, 20, 30]
        train_losses = experiment_data[num_features]["losses"]["train"]
        plt.figure()
        plt.plot(epochs, train_losses, marker="o")
        plt.title(f"Training Loss for {feature_str} Features")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(epochs)
        plt.savefig(
            os.path.join(working_dir, f"training_loss_{feature_str}_features.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # Plot training accuracy
    try:
        train_accuracies = experiment_data[num_features]["metrics"]["train"]
        plt.figure()
        plt.plot(epochs, train_accuracies, marker="o")
        plt.title(f"Training Accuracy for {feature_str} Features")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(epochs)
        plt.savefig(
            os.path.join(working_dir, f"training_accuracy_{feature_str}_features.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training accuracy plot: {e}")
        plt.close()
