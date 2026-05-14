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

# Plot training and validation losses for each optimizer and weight decay
for opt_name, weight_dict in experiment_data["optimization_ablation"].items():
    for wd, data in weight_dict.items():
        try:
            epochs = list(range(1, len(data["losses"]["train"]) + 1))
            plt.figure()
            plt.plot(epochs, data["losses"]["train"], label="Training Loss")
            plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
            plt.title(f"Losses for {opt_name} with {wd}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{opt_name}_losses_weight_decay_{wd}.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {opt_name} weight decay {wd}: {e}")
            plt.close()

# Plot WWBI metrics
for opt_name, weight_dict in experiment_data["optimization_ablation"].items():
    for wd, data in weight_dict.items():
        try:
            plt.figure()
            plt.plot(data["metrics"]["val"], marker="o")
            plt.title(f"WWBI Metrics for {opt_name} with {wd}")
            plt.xlabel("Epochs")
            plt.ylabel("WWBI")
            plt.xticks(
                ticks=range(len(data["metrics"]["val"])),
                labels=list(range(1, len(data["metrics"]["val"]) + 1)),
            )
            plt.savefig(
                os.path.join(working_dir, f"{opt_name}_wwbi_weight_decay_{wd}.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating WWBI plot for {opt_name} weight decay {wd}: {e}")
            plt.close()
