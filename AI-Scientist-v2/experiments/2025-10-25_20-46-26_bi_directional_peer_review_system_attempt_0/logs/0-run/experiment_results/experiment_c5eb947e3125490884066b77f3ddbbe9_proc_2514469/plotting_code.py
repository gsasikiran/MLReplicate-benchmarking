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

# Plot training and validation losses
for num_hidden_units in [8, 16, 32]:
    try:
        losses = experiment_data[
            f"hyperparam_tuning_num_hidden_units_{num_hidden_units}"
        ]["feedback_dataset"]["losses"]
        plt.figure()
        plt.plot(losses["train"], label="Training Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f"Loss Curves for {num_hidden_units} Hidden Units")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"loss_curves_{num_hidden_units}_units.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {num_hidden_units} units: {e}")
        plt.close()

# Plot training metrics
for num_hidden_units in [8, 16, 32]:
    try:
        metrics = experiment_data[
            f"hyperparam_tuning_num_hidden_units_{num_hidden_units}"
        ]["feedback_dataset"]["metrics"]["train"]
        plt.figure()
        plt.plot(metrics, label="Training Metric")
        plt.title(f"Training Metric for {num_hidden_units} Hidden Units")
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"training_metric_{num_hidden_units}_units.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {num_hidden_units} units: {e}")
        plt.close()
