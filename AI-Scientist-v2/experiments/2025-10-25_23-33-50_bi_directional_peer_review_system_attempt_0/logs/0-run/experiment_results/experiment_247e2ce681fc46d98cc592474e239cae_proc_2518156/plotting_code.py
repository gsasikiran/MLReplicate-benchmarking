import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()

    # Plotting training and validation loss curves
    plt.figure()
    dropout_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    for dropout in dropout_rates:
        losses_train = experiment_data["dropout_tuning"]["peer_review"]["losses"][
            "train"
        ]
        losses_val = experiment_data["dropout_tuning"]["peer_review"]["losses"]["val"]
        plt.plot(range(len(losses_train)), losses_train, label=f"Train {dropout}")
        plt.plot(
            range(len(losses_val)), losses_val, linestyle="--", label=f"Val {dropout}"
        )
    plt.title("Training and Validation Loss per Dropout Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "peer_review_loss_curves.png"))
    plt.close()

    # Plotting predictions vs ground truth for each dropout rate
    for i, dropout in enumerate(dropout_rates):
        plt.figure()
        predictions = experiment_data["dropout_tuning"]["peer_review"]["predictions"][i]
        ground_truth = experiment_data["dropout_tuning"]["peer_review"]["ground_truth"][
            i
        ]
        plt.scatter(ground_truth, predictions)
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.title(f"Predictions vs Ground Truth (Dropout Rate: {dropout})")
        plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # Perfect prediction line
        plt.savefig(
            os.path.join(working_dir, f"peer_review_predictions_dropout_{dropout}.png")
        )
        plt.close()

except Exception as e:
    print(f"Error loading or plotting experiment data: {e}")
