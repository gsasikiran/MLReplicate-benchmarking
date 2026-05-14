import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    for momentum, data in zip(
        [0.0, 0.5, 0.9],
        experiment_data["hyperparam_tuning_momentum"]["feedback_data"]["losses"][
            "train"
        ],
    ):
        plt.figure()
        plt.plot(data, label="Training Loss")
        plt.plot(
            experiment_data["hyperparam_tuning_momentum"]["feedback_data"]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"Momentum = {momentum} - Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"momentum_{momentum}_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
