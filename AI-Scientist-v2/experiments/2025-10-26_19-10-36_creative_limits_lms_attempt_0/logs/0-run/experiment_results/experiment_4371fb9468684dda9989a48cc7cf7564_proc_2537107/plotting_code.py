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

for seq_length in experiment_data["sequence_length_variation"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["sequence_length_variation"][seq_length]["losses"]["train"],
            label="Training Loss",
        )
        plt.title(f"{seq_length} Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{seq_length}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot for {seq_length}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["sequence_length_variation"][seq_length]["metrics"][
                "train"
            ],
            label="CODS",
        )
        plt.title(f"{seq_length} CODS Metric")
        plt.xlabel("Epochs")
        plt.ylabel("CODS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{seq_length}_cods_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CODS plot for {seq_length}: {e}")
        plt.close()
