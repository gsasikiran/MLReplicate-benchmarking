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
    for hidden_layer_size, data in experiment_data[
        "hyperparam_tuning_hidden_layer_size"
    ].items():
        plt.plot(data["losses"]["train"], label=f"Train {hidden_layer_size}")
        plt.plot(data["losses"]["val"], label=f"Val {hidden_layer_size}")
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()
