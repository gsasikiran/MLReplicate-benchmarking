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

weight_decay_values = [0.0, 0.01, 0.1, 1.0]
epochs = len(
    experiment_data["hyperparam_tuning_weight_decay"]["peer_review"]["losses"]["train"]
)

for i, weight_decay in enumerate(weight_decay_values):
    try:
        plt.figure()
        plt.plot(
            range(1, epochs + 1),
            experiment_data["hyperparam_tuning_weight_decay"]["peer_review"]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.plot(
            range(1, epochs + 1),
            experiment_data["hyperparam_tuning_weight_decay"]["peer_review"]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for Weight Decay: {weight_decay}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"loss_curves_weight_decay_{weight_decay}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for weight decay {weight_decay}: {e}")
        plt.close()
