import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()

    # Scaled Data Losses Plot
    plt.figure()
    plt.plot(
        experiment_data["feature_scaling_ablation"]["scaled_data"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["feature_scaling_ablation"]["scaled_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Scaled Data Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "scaled_data_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating scaled data loss plot: {e}")
    plt.close()

try:
    # Unscaled Data Losses Plot
    plt.figure()
    plt.plot(
        experiment_data["feature_scaling_ablation"]["unscaled_data"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["feature_scaling_ablation"]["unscaled_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Unscaled Data Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "unscaled_data_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating unscaled data loss plot: {e}")
    plt.close()
