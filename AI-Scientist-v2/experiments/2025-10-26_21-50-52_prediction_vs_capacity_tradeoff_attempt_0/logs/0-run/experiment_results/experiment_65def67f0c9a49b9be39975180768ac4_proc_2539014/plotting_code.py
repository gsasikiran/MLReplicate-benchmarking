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

# Plot training loss for original dataset
try:
    plt.figure()
    epochs = [5, 10, 20, 30]
    plt.plot(
        epochs,
        experiment_data["ablation_study_feature_engineering"]["original_dataset"][
            "losses"
        ]["train"],
        label="Train Loss",
    )
    plt.title("Original Dataset Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "original_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for original dataset training loss: {e}")
    plt.close()

# Plot training accuracy for original dataset
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["ablation_study_feature_engineering"]["original_dataset"][
            "metrics"
        ]["train"],
        label="Train Accuracy",
    )
    plt.title("Original Dataset Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "original_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for original dataset training accuracy: {e}")
    plt.close()

# Plot training loss for polynomial dataset
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["ablation_study_feature_engineering"]["polynomial_dataset"][
            "losses"
        ]["train"],
        label="Train Loss",
    )
    plt.title("Polynomial Dataset Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "polynomial_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for polynomial dataset training loss: {e}")
    plt.close()

# Plot training accuracy for polynomial dataset
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["ablation_study_feature_engineering"]["polynomial_dataset"][
            "metrics"
        ]["train"],
        label="Train Accuracy",
    )
    plt.title("Polynomial Dataset Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "polynomial_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for polynomial dataset training accuracy: {e}")
    plt.close()
