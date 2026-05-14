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

for loss_name in experiment_data["ablation_study"]:
    for batch_size in experiment_data["ablation_study"][loss_name]:
        try:
            plt.figure()
            train_losses = experiment_data["ablation_study"][loss_name][batch_size][
                "losses"
            ]["train"]
            val_losses = experiment_data["ablation_study"][loss_name][batch_size][
                "losses"
            ]["val"]
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title(f"Loss Curve - Loss: {loss_name}, Batch Size: {batch_size}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir, f"Loss_Curve_{loss_name}_BatchSize_{batch_size}.png"
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating loss plot for {loss_name} and batch size {batch_size}: {e}"
            )
            plt.close()
