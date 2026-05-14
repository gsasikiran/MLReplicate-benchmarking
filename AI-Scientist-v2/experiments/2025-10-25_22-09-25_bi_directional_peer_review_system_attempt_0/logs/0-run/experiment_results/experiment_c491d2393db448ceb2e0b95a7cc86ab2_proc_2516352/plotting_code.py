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

for batch_size in [16, 32, 64, 128]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.title(f"Training Loss Curve (Batch Size {batch_size})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"training_loss_batch_size_{batch_size}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot for batch size {batch_size}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(
            experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
                "ground_truth"
            ],
            experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"][
                "predictions"
            ],
            alpha=0.5,
        )
        plt.title(f"Predictions vs Ground Truth (Batch Size {batch_size})")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.axline((0, 0), slope=1, color="r", linestyle="--")  # Line y=x
        plt.savefig(
            os.path.join(
                working_dir, f"predictions_vs_ground_truth_batch_size_{batch_size}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for batch size {batch_size}: {e}")
        plt.close()
