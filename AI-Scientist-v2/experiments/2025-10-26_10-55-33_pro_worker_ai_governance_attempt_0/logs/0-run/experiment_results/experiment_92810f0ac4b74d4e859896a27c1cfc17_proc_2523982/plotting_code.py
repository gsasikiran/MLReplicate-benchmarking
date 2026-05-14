import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    epochs = range(
        1,
        len(
            experiment_data["learning_rate_schedule"]["synthetic_dataset"]["losses"][
                "train"
            ]
        )
        + 1,
    )

    plt.figure()
    plt.plot(
        epochs,
        experiment_data["learning_rate_schedule"]["synthetic_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.plot(
        epochs,
        experiment_data["learning_rate_schedule"]["synthetic_dataset"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Loss Curves: Training and Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["learning_rate_schedule"]["synthetic_dataset"]["metrics"][
            "val"
        ],
        label="PWIS",
    )
    plt.title("Validation Performance Weighted Index Score (PWIS)")
    plt.xlabel("Epochs")
    plt.ylabel("PWIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_pw_is_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating PWIS plot: {e}")
    plt.close()
