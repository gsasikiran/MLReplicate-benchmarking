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
    # Plotting training and validation losses
    epochs = np.arange(1, len(experiment_data["synthetic_data"]["losses"]["train"]) + 1)
    plt.figure()
    plt.plot(
        epochs, experiment_data["synthetic_data"]["losses"]["train"], label="Train Loss"
    )
    plt.plot(
        epochs,
        experiment_data["synthetic_data"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_training_validation_losses.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training/validation loss plot: {e}")
    plt.close()

try:
    # Plotting ground truth vs predicted values
    val_outputs = experiment_data["synthetic_data"]["predictions"]
    ground_truth = experiment_data["synthetic_data"]["ground_truth"]
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth, val_outputs, alpha=0.5)
    plt.xlabel("Ground Truth EIS")
    plt.ylabel("Predicted EIS")
    plt.title("Ground Truth vs Predicted EIS")
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_predicted_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predicted plot: {e}")
    plt.close()
