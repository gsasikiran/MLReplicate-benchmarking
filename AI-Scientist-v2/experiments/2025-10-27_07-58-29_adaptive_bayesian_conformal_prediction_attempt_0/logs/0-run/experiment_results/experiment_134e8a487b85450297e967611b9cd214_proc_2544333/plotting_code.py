import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for dim in [1, 3, 5]:
    try:
        train_losses = experiment_data["input_feature_dimensionality_reduction"][
            f"{dim}D"
        ]["losses"]["train"]
        val_losses = experiment_data["input_feature_dimensionality_reduction"][
            f"{dim}D"
        ]["losses"]["val"]

        plt.figure()
        plt.plot(train_losses, label="Training Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="orange")
        plt.title(f"{dim}D Dimensionality Reduction Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dim}D_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dim}D: {e}")
        plt.close()

    try:
        predictions = experiment_data["input_feature_dimensionality_reduction"][
            f"{dim}D"
        ]["predictions"]
        ground_truth = experiment_data["input_feature_dimensionality_reduction"][
            f"{dim}D"
        ]["ground_truth"]

        # Plotting a few generated samples at intervals
        for i in range(0, len(predictions), max(1, len(predictions) // 5)):
            plt.figure()
            plt.scatter(
                ground_truth[i], predictions[i], label="Generated vs Ground Truth"
            )
            plt.title(f"{dim}D Ground Truth vs Generated Samples")
            plt.xlabel("Ground Truth")
            plt.ylabel("Generated Predictions")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dim}D_generated_samples_epoch_{i}.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating generated samples plot for {dim}D: {e}")
        plt.close()
