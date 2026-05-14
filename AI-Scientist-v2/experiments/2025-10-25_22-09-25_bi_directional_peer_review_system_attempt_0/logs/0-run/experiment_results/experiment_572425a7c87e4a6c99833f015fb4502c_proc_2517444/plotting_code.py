import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for dataset_name in experiment_data["multi_synthetic_dataset_performance"]:
    try:
        losses = experiment_data["multi_synthetic_dataset_performance"][dataset_name][
            "losses"
        ]["train"]
        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.title(f"{dataset_name} - Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset_name}: {e}")
        plt.close()
