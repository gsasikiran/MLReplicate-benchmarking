import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training losses for clean dataset
try:
    clean_losses = experiment_data["input_noise_robustness"]["clean_dataset"]["losses"][
        "train"
    ]
    epochs = range(1, len(clean_losses) + 1)
    plt.figure()
    plt.plot(epochs, clean_losses, label="Clean Dataset Loss")
    plt.title("Training Loss - Clean Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_clean_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating clean dataset loss plot: {e}")
    plt.close()

# Plotting training losses for noisy dataset
try:
    noisy_losses = experiment_data["input_noise_robustness"]["noisy_dataset"]["losses"][
        "train"
    ]
    epochs = range(1, len(noisy_losses) + 1)
    plt.figure()
    plt.plot(epochs, noisy_losses, label="Noisy Dataset Loss", color="orange")
    plt.title("Training Loss - Noisy Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_noisy_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating noisy dataset loss plot: {e}")
    plt.close()

# Plotting metrics for clean dataset
try:
    clean_metrics = experiment_data["input_noise_robustness"]["clean_dataset"][
        "metrics"
    ]["train"]
    epochs = range(1, len(clean_metrics) + 1)
    plt.figure()
    plt.plot(epochs, clean_metrics, label="Clean Dataset Metric")
    plt.title("Training Metrics - Clean Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("UE")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_metrics_clean_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating clean dataset metrics plot: {e}")
    plt.close()

# Plotting metrics for noisy dataset
try:
    noisy_metrics = experiment_data["input_noise_robustness"]["noisy_dataset"][
        "metrics"
    ]["train"]
    epochs = range(1, len(noisy_metrics) + 1)
    plt.figure()
    plt.plot(epochs, noisy_metrics, label="Noisy Dataset Metric", color="orange")
    plt.title("Training Metrics - Noisy Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("UE")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_metrics_noisy_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating noisy dataset metrics plot: {e}")
    plt.close()
