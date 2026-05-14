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

try:
    plt.figure()
    train_losses = experiment_data["hyperparam_tuning_num_hidden_units"][
        "synthetic_data"
    ]["losses"]["train"]
    plt.plot(range(len(train_losses)), train_losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    train_metrics = experiment_data["hyperparam_tuning_num_hidden_units"][
        "synthetic_data"
    ]["metrics"]["train"]
    plt.plot(range(len(train_metrics)), train_metrics)
    plt.title("Training RQS Metric over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_rqs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training RQS plot: {e}")
    plt.close()
