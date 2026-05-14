import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path = (
        "/path/to/experiment_data.npy"  # adjust your path accordingly
    )
    all_experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Aggregate results
try:
    losses = []
    cods_metrics = []
    for data in all_experiment_data:
        losses.append(data["synthetic_dataset"]["losses"]["train"])
        cods_metrics.append(data["synthetic_dataset"]["metrics"]["train"])

    # Convert lists to arrays for mean and std calculation
    losses = np.array(losses)
    cods_metrics = np.array(cods_metrics)

    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0) / np.sqrt(losses.shape[0])

    mean_cods = np.mean(cods_metrics, axis=0)
    std_cods = np.std(cods_metrics, axis=0) / np.sqrt(cods_metrics.shape[0])

    # Plot mean training loss with standard error
    plt.figure()
    epochs = np.arange(len(mean_losses))
    plt.plot(epochs, mean_losses, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_losses - std_losses,
        mean_losses + std_losses,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Mean Training Loss plot: {e}")
    plt.close()

# Plot mean CODS metric with standard error
try:
    plt.figure()
    plt.plot(epochs, mean_cods, label="Mean Train CODS")
    plt.fill_between(
        epochs,
        mean_cods - std_cods,
        mean_cods + std_cods,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Mean Train CODS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_mean_train_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Mean Train CODS plot: {e}")
    plt.close()
