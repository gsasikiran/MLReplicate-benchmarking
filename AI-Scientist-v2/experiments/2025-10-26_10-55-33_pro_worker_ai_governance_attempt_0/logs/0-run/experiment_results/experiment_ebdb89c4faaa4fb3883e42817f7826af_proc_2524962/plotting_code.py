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

for activation_name in experiment_data["ablation_activation_functions"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_activation_functions"][activation_name]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["ablation_activation_functions"][activation_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for {activation_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{activation_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {activation_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_activation_functions"][activation_name][
                "metrics"
            ]["val"],
            label="PWIS",
        )
        plt.title(f"PWIS Metric for {activation_name}")
        plt.xlabel("Epoch")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"PWIS_{activation_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PWIS plot for {activation_name}: {e}")
        plt.close()
