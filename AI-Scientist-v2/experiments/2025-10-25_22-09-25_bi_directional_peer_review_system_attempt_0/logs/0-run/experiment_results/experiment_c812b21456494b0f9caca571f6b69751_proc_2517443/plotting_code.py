import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")


# Function to plot metrics and losses
def plot_experiment_data(experiment_data):
    for key in experiment_data["ablation_study"]:
        try:
            plt.figure()
            plt.plot(
                experiment_data["ablation_study"][key]["losses"]["train"],
                label="Training Loss",
            )
            plt.title(f'{key.replace("_", " ").title()} - Training Loss')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{key}_training_loss.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating {key} training loss plot: {e}")

    try:
        plt.figure()
        plt.plot(
            experiment_data["ablation_study"]["full_dataset"]["metrics"]["train"],
            label="RQS - Full Dataset",
        )
        plt.title("Full Dataset - Training RQS")
        plt.xlabel("Epochs")
        plt.ylabel("RQS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "full_dataset_training_rqs.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Full Dataset RQS plot: {e}")


plot_experiment_data(experiment_data)
