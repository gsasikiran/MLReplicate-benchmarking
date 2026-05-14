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

for difficulty in ["easy", "medium", "hard"]:
    try:
        train_losses = experiment_data["multi_dataset_evaluation"][difficulty][
            "losses"
        ]["train"]
        val_losses = experiment_data["multi_dataset_evaluation"][difficulty]["losses"][
            "val"
        ]

        # Calculate mean and standard error
        train_mean = np.mean(train_losses, axis=0)
        val_mean = np.mean(val_losses, axis=0)
        train_se = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        val_se = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

        plt.figure()
        epochs = np.arange(train_mean.size)
        plt.plot(epochs, train_mean, label="Training Loss")
        plt.plot(epochs, val_mean, label="Validation Loss")
        plt.fill_between(
            epochs, train_mean - train_se, train_mean + train_se, alpha=0.2
        )
        plt.fill_between(epochs, val_mean - val_se, val_mean + val_se, alpha=0.2)
        plt.title(f"{difficulty.capitalize()} Sudoku Loss Curves with Mean and SE")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{difficulty}_sudoku_loss_curves_mean_se.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {difficulty} difficulty: {e}")
        plt.close()
