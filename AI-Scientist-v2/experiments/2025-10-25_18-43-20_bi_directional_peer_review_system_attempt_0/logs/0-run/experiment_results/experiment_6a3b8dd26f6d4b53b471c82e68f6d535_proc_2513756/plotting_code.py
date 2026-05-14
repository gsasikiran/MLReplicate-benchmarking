import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    # Load experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training and validation losses
for num_epochs in [10, 50, 100]:
    try:
        train_losses = experiment_data["training_epochs_impact"][
            f"{num_epochs}_epochs"
        ]["losses"]["train"]
        val_losses = experiment_data["training_epochs_impact"][f"{num_epochs}_epochs"][
            "losses"
        ]["val"]
        epochs = range(len(train_losses))

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss over Epochs for {num_epochs} Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"loss_over_epochs_{num_epochs}_epochs.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {num_epochs} epochs: {e}")
        plt.close()
