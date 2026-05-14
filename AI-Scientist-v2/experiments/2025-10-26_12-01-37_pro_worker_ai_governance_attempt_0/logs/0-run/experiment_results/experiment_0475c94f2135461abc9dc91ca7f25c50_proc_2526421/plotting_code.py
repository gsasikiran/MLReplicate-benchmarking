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

for noise_level in ["low_noise", "medium_noise", "high_noise"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][noise_level]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["ablation_study"][noise_level]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f'Loss Curves for {noise_level.replace("_", " ").title()} Dataset')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{noise_level}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {noise_level}: {e}")
        plt.close()  # Always close figure even if error occurs

    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][noise_level]["metrics"]["val"],
            label="EIS (Validation)",
        )
        plt.title(f'EIS Metric for {noise_level.replace("_", " ").title()} Dataset')
        plt.xlabel("Epochs")
        plt.ylabel("EIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"eis_{noise_level}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating EIS plot for {noise_level}: {e}")
        plt.close()
