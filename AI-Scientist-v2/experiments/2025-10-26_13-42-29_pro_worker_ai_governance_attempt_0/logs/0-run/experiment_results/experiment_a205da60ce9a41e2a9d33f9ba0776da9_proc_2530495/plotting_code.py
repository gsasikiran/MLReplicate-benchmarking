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

for wd in experiment_data["input_noise_ablation"]:
    try:
        train_losses = experiment_data["input_noise_ablation"][wd]["losses"]["train"]
        val_losses = experiment_data["input_noise_ablation"][wd]["losses"]["val"]
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss Curves (Weight Decay: {wd})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_weight_decay_{wd}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for weight decay {wd}: {e}")
        plt.close()
