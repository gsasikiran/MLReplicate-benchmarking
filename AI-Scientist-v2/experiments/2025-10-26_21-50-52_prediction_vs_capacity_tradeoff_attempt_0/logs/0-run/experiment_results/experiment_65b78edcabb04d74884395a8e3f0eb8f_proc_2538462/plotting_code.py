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

for batch_size in [16, 32, 64]:
    try:
        train_losses = experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
            "losses"
        ]["train"]
        epochs = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.title(f"Batch Size {batch_size} - Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"batch_size_{batch_size}_training_loss.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {batch_size} training loss: {e}")

    try:
        train_metrics = experiment_data["batch_size_tuning"][
            f"batch_size_{batch_size}"
        ]["metrics"]["train"]
        epochs = range(1, len(train_metrics) + 1)

        plt.figure()
        plt.plot(epochs, train_metrics, label="Training Accuracy", color="green")
        plt.title(f"Batch Size {batch_size} - Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"batch_size_{batch_size}_training_accuracy.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for batch size {batch_size} training accuracy: {e}")
