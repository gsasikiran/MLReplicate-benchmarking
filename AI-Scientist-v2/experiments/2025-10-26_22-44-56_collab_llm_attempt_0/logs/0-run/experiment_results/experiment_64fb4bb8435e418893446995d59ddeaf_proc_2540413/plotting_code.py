import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

datasets = ["direct_answer", "elaborated_explanation", "conversational_reply"]

for dataset_name in datasets:
    try:
        plt.figure()
        losses = experiment_data["multi_dataset_generalization"][dataset_name][
            "losses"
        ]["train"]
        plt.plot(range(len(losses)), losses, label="Training Loss")
        plt.title(f"{dataset_name.capitalize()} Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {dataset_name} training loss plot: {e}")
        plt.close()
