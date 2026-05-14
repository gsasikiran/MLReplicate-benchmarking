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

# Plot training and validation loss
try:
    epochs = range(
        1,
        len(
            experiment_data["input_noise_injection"]["FeedbackDataset"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    train_loss = experiment_data["input_noise_injection"]["FeedbackDataset"]["losses"][
        "train"
    ]
    val_loss = experiment_data["input_noise_injection"]["FeedbackDataset"]["losses"][
        "val"
    ]

    plt.figure()
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss Curves for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Optionally, add more plots following similar try-except blocks...
