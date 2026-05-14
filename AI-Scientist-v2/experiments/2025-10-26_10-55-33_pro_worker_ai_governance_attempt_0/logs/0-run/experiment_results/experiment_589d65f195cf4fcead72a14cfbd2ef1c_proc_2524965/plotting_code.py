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

for init_method in experiment_data["varying_model_initialization"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["varying_model_initialization"][init_method]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["varying_model_initialization"][init_method]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for Initialization: {init_method}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{init_method}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for loss curves ({init_method}): {e}")
        plt.close()
