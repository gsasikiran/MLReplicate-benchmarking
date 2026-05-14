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

activation_names = experiment_data["activation_function_ablation"].keys()

for activation_name in activation_names:
    try:
        plt.figure()
        plt.plot(
            experiment_data["activation_function_ablation"][activation_name]["metrics"][
                "train"
            ],
            label="Train Accuracy",
        )
        plt.plot(
            experiment_data["activation_function_ablation"][activation_name]["metrics"][
                "val"
            ],
            label="Validation Accuracy",
        )
        plt.title(f"Accuracy Curve for {activation_name} Activation Function")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"accuracy_curve_{activation_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {activation_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["activation_function_ablation"][activation_name]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["activation_function_ablation"][activation_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"Loss Curve for {activation_name} Activation Function")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curve_{activation_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {activation_name}: {e}")
        plt.close()
