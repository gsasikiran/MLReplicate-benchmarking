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

try:
    plt.figure()
    train_losses = experiment_data["ablation_model_architecture"]["synthetic_dataset"][
        "losses"
    ]["train"]
    plt.plot(train_losses)
    plt.title("Training Loss Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    train_accuracies = experiment_data["ablation_model_architecture"][
        "synthetic_dataset"
    ]["metrics"]["train"]
    plt.plot(train_accuracies)
    plt.title("Training Accuracy Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
