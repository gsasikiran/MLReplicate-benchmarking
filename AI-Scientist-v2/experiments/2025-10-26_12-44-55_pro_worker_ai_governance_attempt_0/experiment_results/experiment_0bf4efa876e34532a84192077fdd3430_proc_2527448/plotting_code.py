import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses for each activation function
for act_name in experiment_data["hyperparam_tuning_activation_function"]:
    try:
        losses = experiment_data["hyperparam_tuning_activation_function"][act_name][
            "losses"
        ]
        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {act_name} Activation Function")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{act_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {act_name}: {e}")
        plt.close()
