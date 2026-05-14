import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training loss for each dropout rate
for i, dropout_rate in enumerate([0.0, 0.2, 0.5, 0.7]):
    try:
        plt.figure()
        plt.plot(
            experiment_data["hyperparam_tuning_dropout"]["synthetic_data"]["losses"][
                "train"
            ],
            label=f"Dropout {dropout_rate}",
        )
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"training_loss_dropout_{dropout_rate}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot for dropout {dropout_rate}: {e}")
        plt.close()

# Plot training metrics (UES) for each dropout rate
for i, dropout_rate in enumerate([0.0, 0.2, 0.5, 0.7]):
    try:
        plt.figure()
        plt.plot(
            experiment_data["hyperparam_tuning_dropout"]["synthetic_data"]["metrics"][
                "train"
            ],
            label=f"Dropout {dropout_rate}",
        )
        plt.title("Training UES Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"training_ues_dropout_{dropout_rate}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training UES plot for dropout {dropout_rate}: {e}")
        plt.close()
