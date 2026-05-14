import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    try:
        plt.figure()
        train_losses = experiment_data["hyperparam_tuning_lr"][f"lr_{lr}"]["losses"][
            "train"
        ]
        val_losses = experiment_data["hyperparam_tuning_lr"][f"lr_{lr}"]["losses"][
            "val"
        ]
        epochs = np.arange(len(train_losses))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Loss Curves for Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for lr={lr}: {e}")
        plt.close()

    try:
        plt.figure()
        predictions = experiment_data["hyperparam_tuning_lr"][f"lr_{lr}"]["predictions"]
        ground_truth = experiment_data["hyperparam_tuning_lr"][f"lr_{lr}"][
            "ground_truth"
        ]
        for i in range(0, len(predictions), len(predictions) // 5):
            plt.scatter(ground_truth, predictions[i], label=f"Epoch {i+1}")
        plt.title(f"Predictions vs Ground Truth for Learning Rate: {lr}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"predictions_vs_gt_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for lr={lr}: {e}")
        plt.close()
