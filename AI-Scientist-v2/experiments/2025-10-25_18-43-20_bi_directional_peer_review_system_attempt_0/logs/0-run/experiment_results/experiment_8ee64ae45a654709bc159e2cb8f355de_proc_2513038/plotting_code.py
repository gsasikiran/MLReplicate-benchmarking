import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot Training Loss
try:
    train_losses = experiment_data["hyperparam_tuning_additional_hidden_layer"][
        "RQI_experiment"
    ]["losses"]["train"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQI_experiment_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot Validation Loss
try:
    val_losses = experiment_data["hyperparam_tuning_additional_hidden_layer"][
        "RQI_experiment"
    ]["losses"]["val"]
    plt.figure()
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQI_experiment_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()
