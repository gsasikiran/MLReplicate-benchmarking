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

for idx in range(2):
    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_activation_function_comparison"][
                f"activation_combination_{idx + 1}"
            ]["losses"]["train"],
            label="Training Loss",
        )
        plt.title(f"Activation Combination {idx + 1}: Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"activation_combination_{idx + 1}_training_loss.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot for combination {idx + 1}: {e}")
        plt.close()

for idx in range(2):
    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_activation_function_comparison"][
                f"activation_combination_{idx + 1}"
            ]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Activation Combination {idx + 1}: Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"activation_combination_{idx + 1}_validation_loss.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot for combination {idx + 1}: {e}")
        plt.close()
