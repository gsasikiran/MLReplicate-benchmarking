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

for missing_rate in experiment_data["multiple_synthetic_datasets"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][missing_rate]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][missing_rate]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{missing_rate}: Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{missing_rate}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot for {missing_rate}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][missing_rate]["mdie"][
                "train"
            ],
            label="Train MDIE",
        )
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][missing_rate]["mdie"]["val"],
            label="Validation MDIE",
        )
        plt.title(f"{missing_rate}: Training and Validation MDIE")
        plt.xlabel("Epochs")
        plt.ylabel("MDIE")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{missing_rate}_mdie_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MDIE plot for {missing_rate}: {e}")
        plt.close()

    # Generating synthetic samples plot can be added similarly if needed.
