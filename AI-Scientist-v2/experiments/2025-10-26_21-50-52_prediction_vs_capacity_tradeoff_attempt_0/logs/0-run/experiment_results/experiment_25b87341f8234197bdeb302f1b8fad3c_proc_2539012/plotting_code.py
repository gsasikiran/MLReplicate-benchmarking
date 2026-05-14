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

try:
    train_losses = experiment_data["ensemble_model_variation"]["synthetic_dataset"][
        "losses"
    ]["train"]
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    train_accuracies = experiment_data["ensemble_model_variation"]["synthetic_dataset"][
        "metrics"
    ]["train"]
    plt.figure()
    plt.plot(train_accuracies)
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
