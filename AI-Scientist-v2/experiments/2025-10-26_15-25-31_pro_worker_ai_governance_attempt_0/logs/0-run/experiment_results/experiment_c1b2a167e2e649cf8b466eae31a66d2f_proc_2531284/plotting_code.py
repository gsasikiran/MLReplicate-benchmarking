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
    plt.plot(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_data"][
            "losses"
        ]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_data"][
            "losses"
        ]["val"],
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_training_validation_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_data"][
            "metrics"
        ]["train"],
        label="Train Accuracy",
    )
    plt.plot(
        experiment_data["hyperparam_tuning_hidden_layer_size"]["synthetic_data"][
            "metrics"
        ]["val"],
        label="Validation Accuracy",
    )
    plt.title("Training and Validation Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_training_validation_accuracy.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
