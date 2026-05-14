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
    unscaled_losses = experiment_data["Input_Feature_Scaling_Impact"]["unscaled_data"][
        "losses"
    ]["train"]
    scaled_losses = experiment_data["Input_Feature_Scaling_Impact"]["scaled_data"][
        "losses"
    ]["train"]

    epochs = np.arange(len(unscaled_losses))

    # Calculate mean and standard error
    unscaled_mean = np.mean(unscaled_losses)
    unscaled_se = np.std(unscaled_losses) / np.sqrt(len(unscaled_losses))

    scaled_mean = np.mean(scaled_losses)
    scaled_se = np.std(scaled_losses) / np.sqrt(len(scaled_losses))

    plt.figure()
    plt.errorbar(
        epochs, unscaled_losses, yerr=unscaled_se, label="Unscaled Loss", capsize=5
    )
    plt.title("Input Feature Scaling Impact")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(
        ticks=np.arange(0, len(unscaled_losses), max(1, len(unscaled_losses) // 5))
    )  # At most 5 ticks
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Loss_Unscaled_with_SE.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for unscaled training loss: {e}")
    plt.close()

try:
    plt.figure()
    plt.errorbar(
        epochs,
        scaled_losses,
        yerr=scaled_se,
        label="Scaled Loss",
        color="orange",
        capsize=5,
    )
    plt.title("Input Feature Scaling Impact")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(
        ticks=np.arange(0, len(scaled_losses), max(1, len(scaled_losses) // 5))
    )  # At most 5 ticks
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Loss_Scaled_with_SE.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for scaled training loss: {e}")
    plt.close()
