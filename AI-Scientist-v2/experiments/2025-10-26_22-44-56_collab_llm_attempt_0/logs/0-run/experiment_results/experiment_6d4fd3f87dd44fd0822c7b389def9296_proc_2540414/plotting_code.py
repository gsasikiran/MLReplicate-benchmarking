import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training losses
for dataset_type in experiment_data["multi_dataset_evaluation"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][dataset_type]["losses"]["train"]
        )
        plt.title(f"Training Losses for {dataset_type}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(working_dir, f"{dataset_type}_training_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training losses plot for {dataset_type}: {e}")
        plt.close()

# Plot training metrics (UES)
for dataset_type in experiment_data["multi_dataset_evaluation"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][dataset_type]["metrics"][
                "train"
            ]
        )
        plt.title(f"Training Metrics (UES) for {dataset_type}")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.savefig(os.path.join(working_dir, f"{dataset_type}_training_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {dataset_type}: {e}")
        plt.close()

# Plot predictions vs ground truth for each dataset
for dataset_type in experiment_data["multi_dataset_evaluation"]:
    try:
        pred = np.array(
            experiment_data["multi_dataset_evaluation"][dataset_type]["predictions"]
        )
        gt = np.array(
            experiment_data["multi_dataset_evaluation"][dataset_type]["ground_truth"]
        )

        plt.figure()
        plt.scatter(gt[:, 0], gt[:, 1], label="Ground Truth", color="blue", alpha=0.5)
        plt.scatter(pred[:, 0], pred[:, 1], label="Predictions", color="red", alpha=0.5)
        plt.title(f"Predictions vs Ground Truth for {dataset_type}")
        plt.xlabel("Value 1")
        plt.ylabel("Value 2")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_type}_predictions_vs_truth.png")
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating predictions vs ground truth plot for {dataset_type}: {e}"
        )
        plt.close()
