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

# Plot training and validation loss for fixed learning rate
try:
    fixed_train_losses = experiment_data["learning_rate_schedule"]["fixed"]["losses"][
        "train"
    ]
    fixed_val_losses = experiment_data["learning_rate_schedule"]["fixed"]["losses"][
        "val"
    ]
    plt.figure()
    plt.plot(fixed_train_losses, label="Train Loss")
    plt.plot(fixed_val_losses, label="Validation Loss")
    plt.title("Fixed Learning Rate Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "fixed_lr_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating fixed learning rate loss curve: {e}")
    plt.close()

# Plot training and validation loss for scheduled learning rate
try:
    schedule_train_losses = experiment_data["learning_rate_schedule"]["schedule"][
        "losses"
    ]["train"]
    schedule_val_losses = experiment_data["learning_rate_schedule"]["schedule"][
        "losses"
    ]["val"]
    plt.figure()
    plt.plot(schedule_train_losses, label="Train Loss")
    plt.plot(schedule_val_losses, label="Validation Loss")
    plt.title("Scheduled Learning Rate Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "scheduled_lr_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating scheduled learning rate loss curve: {e}")
    plt.close()

# Plot predictions vs ground truth for fixed learning rate
try:
    fixed_predictions = experiment_data["learning_rate_schedule"]["fixed"][
        "predictions"
    ]
    fixed_ground_truth = experiment_data["learning_rate_schedule"]["fixed"][
        "ground_truth"
    ]
    plt.figure()
    plt.scatter(fixed_ground_truth, fixed_predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], "--r")  # Line for reference
    plt.title("Fixed LR Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "fixed_lr_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating fixed LR predictions plot: {e}")
    plt.close()

# Plot predictions vs ground truth for scheduled learning rate
try:
    schedule_predictions = experiment_data["learning_rate_schedule"]["schedule"][
        "predictions"
    ]
    schedule_ground_truth = experiment_data["learning_rate_schedule"]["schedule"][
        "ground_truth"
    ]
    plt.figure()
    plt.scatter(schedule_ground_truth, schedule_predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], "--r")  # Line for reference
    plt.title("Scheduled LR Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "scheduled_lr_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating scheduled LR predictions plot: {e}")
    plt.close()
