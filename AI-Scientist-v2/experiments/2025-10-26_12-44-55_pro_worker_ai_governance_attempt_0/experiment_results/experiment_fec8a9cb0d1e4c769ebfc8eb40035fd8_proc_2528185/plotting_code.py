import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses for original dataset
try:
    plt.figure()
    plt.plot(
        experiment_data["multi_feature_interaction"]["original_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["multi_feature_interaction"]["original_dataset"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Losses for Original Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "original_dataset_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating original dataset loss plot: {e}")
    plt.close()

# Plot WIS for original dataset
try:
    plt.figure()
    plt.plot(
        experiment_data["multi_feature_interaction"]["original_dataset"]["metrics"][
            "val"
        ],
        label="WIS",
    )
    plt.title("WIS for Original Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("WIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "original_dataset_WIS.png"))
    plt.close()
except Exception as e:
    print(f"Error creating original dataset WIS plot: {e}")
    plt.close()

# Plot training and validation losses for interaction dataset
try:
    plt.figure()
    plt.plot(
        experiment_data["multi_feature_interaction"]["interaction_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        experiment_data["multi_feature_interaction"]["interaction_dataset"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Losses for Interaction Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "interaction_dataset_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating interaction dataset loss plot: {e}")
    plt.close()

# Plot WIS for interaction dataset
try:
    plt.figure()
    plt.plot(
        experiment_data["multi_feature_interaction"]["interaction_dataset"]["metrics"][
            "val"
        ],
        label="WIS",
    )
    plt.title("WIS for Interaction Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("WIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "interaction_dataset_WIS.png"))
    plt.close()
except Exception as e:
    print(f"Error creating interaction dataset WIS plot: {e}")
    plt.close()
