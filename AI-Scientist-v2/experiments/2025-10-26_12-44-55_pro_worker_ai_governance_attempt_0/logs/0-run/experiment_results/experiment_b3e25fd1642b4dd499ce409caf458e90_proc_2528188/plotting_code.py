import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot loss curves for each model
for model_name in ["SimpleNN", "DeepNN3", "DeepNN4"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][model_name]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["ablation_study"][model_name]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{model_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {model_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"][model_name]["metrics"]["val"],
            label="Validation Metric (WIS)",
        )
        plt.title(f"{model_name} Validation Metric Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("WIS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{model_name}_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {model_name}: {e}")
        plt.close()
