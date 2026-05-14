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

for model_type in ["original_features", "interaction_feature"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["feature_interaction"][model_type]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["feature_interaction"][model_type]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{model_type.replace('_', ' ').title()} Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_type}_loss_over_epochs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {model_type} losses: {e}")
        plt.close()

    try:
        plt.figure()
        PWIS = [
            1 - val
            for val in experiment_data["feature_interaction"][model_type]["metrics"][
                "val"
            ]
        ]
        plt.plot(PWIS, label="PWIS (Higher is Better)")
        plt.title(f"{model_type.replace('_', ' ').title()} PWIS Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("PWIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_type}_pwis_over_epochs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {model_type} PWIS: {e}")
        plt.close()
