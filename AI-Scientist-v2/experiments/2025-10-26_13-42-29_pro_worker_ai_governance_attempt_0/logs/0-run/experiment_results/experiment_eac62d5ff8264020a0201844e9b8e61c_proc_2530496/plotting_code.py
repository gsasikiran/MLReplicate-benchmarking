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

for act_name in experiment_data["activation_function_ablation"]:
    for wd in experiment_data["activation_function_ablation"][act_name]:
        try:
            plt.figure()
            train_losses = experiment_data["activation_function_ablation"][act_name][
                wd
            ]["losses"]["train"]
            val_losses = experiment_data["activation_function_ablation"][act_name][wd][
                "losses"
            ]["val"]
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title(f"Training and Validation Loss - {act_name}, Weight Decay {wd}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"loss_plot_{act_name}_wd_{wd}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {act_name}, {wd}: {e}")
            plt.close()

        try:
            plt.figure()
            val_metrics = experiment_data["activation_function_ablation"][act_name][wd][
                "metrics"
            ]["val"]
            plt.plot(val_metrics, label="WWBI Metric")
            plt.title(f"WWBI Metric - {act_name}, Weight Decay {wd}")
            plt.xlabel("Epochs")
            plt.ylabel("WWBI")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"wwbi_plot_{act_name}_wd_{wd}.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating WWBI plot for {act_name}, {wd}: {e}")
            plt.close()
