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

for act_name in experiment_data["activation_function_tuning"]:
    try:
        train_losses = experiment_data["activation_function_tuning"][act_name][
            "losses"
        ]["train"]
        val_losses = experiment_data["activation_function_tuning"][act_name]["losses"][
            "val"
        ]

        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
        plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
        plt.title(f"{act_name} Activation Function Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name}: {e}")
        plt.close()
