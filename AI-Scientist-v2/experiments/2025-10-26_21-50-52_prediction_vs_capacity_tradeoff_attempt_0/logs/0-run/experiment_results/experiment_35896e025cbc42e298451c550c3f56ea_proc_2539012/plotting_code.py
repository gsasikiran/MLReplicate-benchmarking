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
    # Plotting training loss
    temperatures = [1.0, 0.5, 2.0]
    for temp in temperatures:
        losses = experiment_data["output_layer_variation"]["synthetic_dataset"][
            "losses"
        ]["train"]
        plt.plot(losses, label=f"Temperature {temp}")
    plt.title("Training Loss per Temperature")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_curves_synthetic_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Plotting training accuracy
    for temp in temperatures:
        accuracy = experiment_data["output_layer_variation"]["synthetic_dataset"][
            "metrics"
        ]["train"]
        plt.plot(accuracy, label=f"Temperature {temp}")
    plt.title("Training Accuracy per Temperature")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "training_accuracy_curves_synthetic_dataset.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
