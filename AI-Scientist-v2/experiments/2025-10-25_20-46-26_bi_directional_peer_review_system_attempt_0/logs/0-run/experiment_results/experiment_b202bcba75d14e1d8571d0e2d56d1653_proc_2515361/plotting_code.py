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
    plt.figure()
    epochs = range(
        1,
        len(
            experiment_data["variability_of_input_features"]["FeedbackDataset"][
                "losses"
            ]["train"]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["variability_of_input_features"]["FeedbackDataset"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["variability_of_input_features"]["FeedbackDataset"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Loss Curves for FeedbackDataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "FeedbackDataset_Loss_Curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()
