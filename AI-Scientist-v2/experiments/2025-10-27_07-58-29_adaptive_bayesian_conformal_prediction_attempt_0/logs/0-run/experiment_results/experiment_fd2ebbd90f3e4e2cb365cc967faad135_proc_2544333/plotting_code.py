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

for scale_type in experiment_data["feature_scale_investigation"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["feature_scale_investigation"][scale_type]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["feature_scale_investigation"][scale_type]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Curves for {scale_type} Scaling")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_curves_{scale_type}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {scale_type}: {e}")
        plt.close()

    try:
        plt.figure()
        val_metrics = experiment_data["feature_scale_investigation"][scale_type][
            "metrics"
        ]["val"]
        plt.plot(val_metrics, label="Validation Reliability", marker="o")
        plt.title(f"Validation Reliability for {scale_type} Scaling")
        plt.xlabel("Epochs")
        plt.ylabel("Reliability Measure")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"reliability_metric_{scale_type}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating reliability metric plot for {scale_type}: {e}")
        plt.close()
