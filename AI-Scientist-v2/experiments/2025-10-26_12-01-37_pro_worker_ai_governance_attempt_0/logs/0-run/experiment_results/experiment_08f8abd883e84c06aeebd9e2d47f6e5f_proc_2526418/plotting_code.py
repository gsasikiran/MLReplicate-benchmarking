import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    for hidden_units in experiment_data["learning_rate_variation"]:
        plt.plot(
            experiment_data["learning_rate_variation"][hidden_units]["losses"]["train"],
            label=f"{hidden_units} Train",
        )
        plt.plot(
            experiment_data["learning_rate_variation"][hidden_units]["losses"]["val"],
            label=f"{hidden_units} Validation",
        )
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Validation_Losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training and Validation Loss plot: {e}")
    plt.close()

try:
    plt.figure()
    predictions_fixed = experiment_data["learning_rate_variation"]["fixed_lr"][
        "predictions"
    ]
    ground_truth_fixed = experiment_data["learning_rate_variation"]["fixed_lr"][
        "ground_truth"
    ]
    plt.scatter(ground_truth_fixed, predictions_fixed, alpha=0.5)
    plt.plot([0, 1], [0, 1], "r--")
    plt.title("Predictions vs Ground Truth (Fixed Learning Rate)")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(os.path.join(working_dir, "Predictions_vs_Ground_Truth_Fixed_LR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Predictions vs Ground Truth Fixed LR plot: {e}")
    plt.close()

try:
    plt.figure()
    predictions_scheduler = experiment_data["learning_rate_variation"]["scheduler_lr"][
        "predictions"
    ]
    ground_truth_scheduler = experiment_data["learning_rate_variation"]["scheduler_lr"][
        "ground_truth"
    ]
    plt.scatter(ground_truth_scheduler, predictions_scheduler, alpha=0.5)
    plt.plot([0, 1], [0, 1], "r--")
    plt.title("Predictions vs Ground Truth (Scheduled Learning Rate)")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(
        os.path.join(working_dir, "Predictions_vs_Ground_Truth_Scheduler_LR.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating Predictions vs Ground Truth Scheduler LR plot: {e}")
    plt.close()
