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

activations = experiment_data["activation_function_comparison"]

for activation_name in activations:
    for batch_size in activations[activation_name]["batch_size_tuning"]:
        try:
            losses = activations[activation_name]["batch_size_tuning"][batch_size][
                "losses"
            ]
            epochs = list(range(len(losses["train"])))
            plt.figure()
            plt.plot(epochs, losses["train"], label="Train Loss")
            plt.plot(epochs, losses["val"], label="Validation Loss")
            plt.title(
                f"{activation_name} Activation Function - Batch Size {batch_size}"
            )
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir, f"loss_curve_{activation_name}_batch_{batch_size}.png"
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating loss curve for {activation_name}, batch {batch_size}: {e}"
            )
            plt.close()

        try:
            metrics = activations[activation_name]["batch_size_tuning"][batch_size][
                "metrics"
            ]
            PWIS_values = metrics["val"]
            plt.figure()
            plt.plot(PWIS_values, label="PWIS")
            plt.title(
                f"{activation_name} Activation Function - PWIS for Batch Size {batch_size}"
            )
            plt.xlabel("Epochs")
            plt.ylabel("PWIS")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir, f"PWIS_curve_{activation_name}_batch_{batch_size}.png"
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating PWIS curve for {activation_name}, batch {batch_size}: {e}"
            )
            plt.close()
