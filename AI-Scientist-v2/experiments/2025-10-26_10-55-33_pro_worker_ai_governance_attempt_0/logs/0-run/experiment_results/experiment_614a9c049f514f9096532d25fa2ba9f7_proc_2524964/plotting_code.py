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

for model_key in experiment_data["varying_model_complexity"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["varying_model_complexity"][model_key]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["varying_model_complexity"][model_key]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f'Loss Curves for {model_key.replace("_", " ").capitalize()} Model')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_key}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {model_key}: {e}")
        plt.close()
