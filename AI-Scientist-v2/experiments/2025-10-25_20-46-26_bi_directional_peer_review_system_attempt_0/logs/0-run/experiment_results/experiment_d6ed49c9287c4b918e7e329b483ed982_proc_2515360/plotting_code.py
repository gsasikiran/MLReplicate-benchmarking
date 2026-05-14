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

for dataset_index in range(1, 4):
    try:
        dataset_key = f"FeedbackDataset_{dataset_index}"
        train_losses = experiment_data["multi_dataset_evaluation"][dataset_key][
            "losses"
        ]["train"]
        val_losses = experiment_data["multi_dataset_evaluation"][dataset_key]["losses"][
            "val"
        ]
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"{dataset_key} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_key}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_key}: {e}")
        plt.close()

    try:
        train_metrics = experiment_data["multi_dataset_evaluation"][dataset_key][
            "metrics"
        ]["train"]
        epochs = np.arange(1, len(train_metrics) + 1)

        plt.figure()
        plt.plot(epochs, train_metrics, label="Training Metrics")
        plt.title(f"{dataset_key} Training Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_key}_metrics_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {dataset_key}: {e}")
        plt.close()
