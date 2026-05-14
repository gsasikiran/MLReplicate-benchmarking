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

# Plotting function for loss
for dataset in experiment_data["dataset_diversity_ablation"]:
    for wd in experiment_data["dataset_diversity_ablation"][dataset]:
        train_losses = experiment_data["dataset_diversity_ablation"][dataset][wd][
            "losses"
        ]["train"]
        val_losses = experiment_data["dataset_diversity_ablation"][dataset][wd][
            "losses"
        ]["val"]

        try:
            plt.figure()
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, label="Training Loss")
            plt.plot(epochs, val_losses, label="Validation Loss")
            plt.title(f"{dataset} - Weight Decay {wd} Loss Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_wd_{wd}_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating plot for {dataset} with weight decay {wd}: {e}")
            plt.close()
