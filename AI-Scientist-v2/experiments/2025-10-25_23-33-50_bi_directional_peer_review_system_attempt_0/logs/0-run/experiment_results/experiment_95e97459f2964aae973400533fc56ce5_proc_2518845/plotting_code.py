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
    # Plotting train and val losses for no interaction
    plt.figure()
    plt.plot(
        experiment_data["feature_interaction"]["no_interaction"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["feature_interaction"]["no_interaction"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves (No Interaction Terms)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "no_interaction_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating no interaction plot: {e}")
    plt.close()

try:
    # Plotting train and val losses for with interaction
    plt.figure()
    plt.plot(
        experiment_data["feature_interaction"]["with_interaction"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["feature_interaction"]["with_interaction"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves (With Interaction Terms)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "with_interaction_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating with interaction plot: {e}")
    plt.close()
