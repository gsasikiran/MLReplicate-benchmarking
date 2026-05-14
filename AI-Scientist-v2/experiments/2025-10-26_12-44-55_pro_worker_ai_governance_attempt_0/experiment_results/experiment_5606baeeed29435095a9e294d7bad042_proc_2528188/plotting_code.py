import matplotlib.pyplot as plt
import numpy as np
import os

# Load experiment data
working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training and validation losses
for split in experiment_data["train_test_split_ratios"]:
    try:
        loss_train = experiment_data["train_test_split_ratios"][split]["losses"][
            "train"
        ]
        loss_val = experiment_data["train_test_split_ratios"][split]["losses"]["val"]
        epochs = range(1, len(loss_train) + 1)

        plt.figure()
        plt.plot(epochs, loss_train, label="Training Loss")
        plt.plot(epochs, loss_val, label="Validation Loss")
        plt.title(f"Loss Curves for Split: {split}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{split}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {split}: {e}")
        plt.close()

# Plotting WIS scores
for split in experiment_data["train_test_split_ratios"]:
    try:
        wis_scores = experiment_data["train_test_split_ratios"][split]["metrics"]["val"]
        epochs = range(1, len(wis_scores) + 1)

        plt.figure()
        plt.plot(epochs, wis_scores, label="WIS Score", color="g")
        plt.title(f"WIS Scores for Split: {split}")
        plt.xlabel("Epochs")
        plt.ylabel("WIS Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"wis_scores_{split}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating WIS plot for {split}: {e}")
        plt.close()
