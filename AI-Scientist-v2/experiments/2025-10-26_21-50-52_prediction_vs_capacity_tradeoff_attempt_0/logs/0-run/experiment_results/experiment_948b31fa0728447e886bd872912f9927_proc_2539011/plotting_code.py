import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for dataset in ["normal_dataset", "uniform_dataset", "skewed_dataset"]:
    try:
        plt.figure()
        epochs = np.arange(
            1,
            len(
                experiment_data["multiple_synthetic_datasets_variation"][dataset][
                    "losses"
                ]["train"]
            )
            + 1,
        )
        plt.plot(
            epochs,
            experiment_data["multiple_synthetic_datasets_variation"][dataset]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.title(f"{dataset.capitalize()} - Training Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset} - Training Loss: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            epochs,
            experiment_data["multiple_synthetic_datasets_variation"][dataset][
                "metrics"
            ]["train"],
            label="Training Accuracy",
        )
        plt.title(f"{dataset.capitalize()} - Training Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_training_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset} - Training Accuracy: {e}")
        plt.close()
