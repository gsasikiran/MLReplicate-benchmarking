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
    train_losses = experiment_data["hyperparam_tuning_num_hidden_units"]["sudoku"][
        "losses"
    ]["train"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss for Sudoku Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    val_losses = experiment_data["hyperparam_tuning_num_hidden_units"]["sudoku"][
        "losses"
    ]["val"]
    plt.figure()
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Validation Loss for Sudoku Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sudoku_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
