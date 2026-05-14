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
    epochs = range(
        1,
        len(
            experiment_data["input_data_normalization"]["mean_imputation"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["input_data_normalization"]["mean_imputation"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["input_data_normalization"]["mean_imputation"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Mean Imputation: Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_imputation_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Mean Imputation plot: {e}")
    plt.close()

try:
    plt.figure()
    epochs = range(
        1,
        len(
            experiment_data["input_data_normalization"]["data_normalization"]["losses"][
                "train"
            ]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["input_data_normalization"]["data_normalization"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        epochs,
        experiment_data["input_data_normalization"]["data_normalization"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Data Normalization: Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "data_normalization_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Data Normalization plot: {e}")
    plt.close()
