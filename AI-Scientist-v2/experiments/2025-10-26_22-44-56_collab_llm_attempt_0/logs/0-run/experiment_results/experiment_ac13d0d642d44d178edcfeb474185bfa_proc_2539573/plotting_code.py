import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training losses
try:
    epochs = np.arange(
        1, len(experiment_data["multi_turn_interactions"]["losses"]["train"]) + 1
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["multi_turn_interactions"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "multi_turn_interactions_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")

# Plot training metrics
try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["multi_turn_interactions"]["metrics"]["train"],
        label="Training UES",
    )
    plt.title("Training UES Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("UES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "multi_turn_interactions_training_ues.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training UES plot: {e}")
