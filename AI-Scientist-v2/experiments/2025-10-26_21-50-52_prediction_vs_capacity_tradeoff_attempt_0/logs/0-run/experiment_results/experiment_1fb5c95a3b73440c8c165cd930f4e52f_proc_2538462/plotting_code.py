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
    # Plotting Training Loss
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"]["losses"][
            "train"
        ]
    )
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Plotting Training Accuracy
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_num_epochs"]["synthetic_dataset"]["metrics"][
            "train"
        ]
    )
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
