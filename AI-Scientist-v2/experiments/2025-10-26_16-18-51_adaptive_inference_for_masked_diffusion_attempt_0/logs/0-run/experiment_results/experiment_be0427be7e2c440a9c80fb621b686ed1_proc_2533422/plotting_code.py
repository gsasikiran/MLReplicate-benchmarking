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

# Plot training and validation losses with normalization
try:
    plt.figure()
    epochs = [5, 10, 15, 20]  # Corresponds with num_epochs_list in the training
    plt.plot(
        epochs,
        experiment_data["input_normalization"]["sudoku"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["input_normalization"]["sudoku"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Sudoku Dataset: Training and Validation Loss with Normalization")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "sudoku_train_val_loss_with_normalization.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot for training/validation loss with normalization: {e}")
    plt.close()

# Plot training and validation losses without normalization
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["no_normalization"]["sudoku"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["no_normalization"]["sudoku"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Sudoku Dataset: Training and Validation Loss without Normalization")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "sudoku_train_val_loss_without_normalization.png")
    )
    plt.close()
except Exception as e:
    print(
        f"Error creating plot for training/validation loss without normalization: {e}"
    )
    plt.close()
