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

# Plot training losses
try:
    plt.figure()
    activation_names = list(
        experiment_data["impact_of_activation_functions"]["synthetic_data"]["losses"][
            "train"
        ].keys()
    )
    losses = [
        experiment_data["impact_of_activation_functions"]["synthetic_data"]["losses"][
            "train"
        ][name]
        for name in activation_names
    ]
    for loss, name in zip(losses, activation_names):
        plt.plot(loss, label=name)
    plt.title("Training Loss per Activation Function")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_per_activation_function.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training RQS metrics
try:
    plt.figure()
    metrics = [
        experiment_data["impact_of_activation_functions"]["synthetic_data"]["metrics"][
            "train"
        ][name]
        for name in activation_names
    ]
    for metric, name in zip(metrics, activation_names):
        plt.plot(metric, label=name)
    plt.title("Training RQS per Activation Function")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_rqs_per_activation_function.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training RQS plot: {e}")
    plt.close()
