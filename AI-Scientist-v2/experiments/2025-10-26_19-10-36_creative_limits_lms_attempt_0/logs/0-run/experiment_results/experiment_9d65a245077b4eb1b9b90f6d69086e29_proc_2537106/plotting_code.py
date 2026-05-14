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
    epochs = list(
        range(
            1,
            len(
                experiment_data["input_noise_variation"]["synthetic_dataset"]["losses"][
                    "train"
                ]
            )
            + 1,
        )
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["input_noise_variation"]["synthetic_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["input_noise_variation"]["synthetic_dataset"]["metrics"][
            "train"
        ],
        label="CODS",
        color="orange",
    )
    plt.title("Training CODS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CODS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_cods.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training CODS plot: {e}")
    plt.close()
