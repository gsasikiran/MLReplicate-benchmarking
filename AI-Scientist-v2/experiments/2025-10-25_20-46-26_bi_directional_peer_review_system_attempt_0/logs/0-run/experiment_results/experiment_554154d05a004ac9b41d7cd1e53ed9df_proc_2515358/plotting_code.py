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

for dataset_name in experiment_data["MultipleSyntheticDatasetEvaluation"]:
    try:
        train_losses = experiment_data["MultipleSyntheticDatasetEvaluation"][
            dataset_name
        ]["losses"]["train"]
        val_losses = experiment_data["MultipleSyntheticDatasetEvaluation"][
            dataset_name
        ]["losses"]["val"]

        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name} loss curves: {e}")
        plt.close()
