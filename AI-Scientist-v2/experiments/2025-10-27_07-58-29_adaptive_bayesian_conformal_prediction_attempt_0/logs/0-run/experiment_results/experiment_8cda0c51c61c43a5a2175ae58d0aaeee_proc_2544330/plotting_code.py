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

for activation in experiment_data["activation_function_exploration"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["activation_function_exploration"][activation]["losses"][
                "train"
            ],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["activation_function_exploration"][activation]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"Loss vs Epochs for {activation}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_plot_{activation}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {activation}: {e}")
        plt.close()

    try:
        predictions = np.array(
            experiment_data["activation_function_exploration"][activation][
                "predictions"
            ]
        ).flatten()
        ground_truth = np.concatenate(
            experiment_data["activation_function_exploration"][activation][
                "ground_truth"
            ]
        )
        plt.figure()
        plt.scatter(ground_truth, predictions)
        plt.plot([0, 3], [0, 3], "r--")  # Ideal line
        plt.title(f"Predictions vs Ground Truth for {activation}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(os.path.join(working_dir, f"predictions_plot_{activation}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for {activation}: {e}")
        plt.close()
