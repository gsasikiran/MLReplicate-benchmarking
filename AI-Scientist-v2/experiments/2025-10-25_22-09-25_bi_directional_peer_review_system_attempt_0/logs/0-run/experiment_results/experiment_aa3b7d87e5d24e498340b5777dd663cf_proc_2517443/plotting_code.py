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

try:
    plt.figure()
    plt.plot(
        experiment_data["Input_Feature_Scaling_Impact"]["unscaled_data"]["losses"][
            "train"
        ],
        label="Unscaled Loss",
    )
    plt.title("Input Feature Scaling Impact")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Loss_Unscaled.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for unscaled training loss: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["Input_Feature_Scaling_Impact"]["scaled_data"]["losses"][
            "train"
        ],
        label="Scaled Loss",
        color="orange",
    )
    plt.title("Input Feature Scaling Impact")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Loss_Scaled.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for scaled training loss: {e}")
    plt.close()
