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
            len(
                experiment_data["impact_of_batch_normalization"]["with_bn"]["losses"][
                    "train"
                ]
            )
        )
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["impact_of_batch_normalization"]["with_bn"]["losses"]["train"],
        label="Train (with BN)",
    )
    plt.plot(
        epochs,
        experiment_data["impact_of_batch_normalization"]["with_bn"]["losses"]["val"],
        label="Validation (with BN)",
    )
    plt.title("Training and Validation Loss with Batch Normalization")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_with_bn.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot (with BN): {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["impact_of_batch_normalization"]["without_bn"]["losses"][
            "train"
        ],
        label="Train (without BN)",
    )
    plt.plot(
        epochs,
        experiment_data["impact_of_batch_normalization"]["without_bn"]["losses"]["val"],
        label="Validation (without BN)",
    )
    plt.title("Training and Validation Loss without Batch Normalization")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_without_bn.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot (without BN): {e}")
    plt.close()
