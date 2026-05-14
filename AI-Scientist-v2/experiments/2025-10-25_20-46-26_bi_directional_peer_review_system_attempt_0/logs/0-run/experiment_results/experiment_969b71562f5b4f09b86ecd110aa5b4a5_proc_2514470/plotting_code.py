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

for dropout_rate, metrics in experiment_data["hyperparam_tuning_dropout"][
    "peer_review_feedback"
]["losses"].items():
    try:
        plt.figure()
        plt.plot(metrics["train"], label="Training Loss", marker="o")
        plt.plot(metrics["val"], label="Validation Loss", marker="x")
        plt.title(f"Loss Curves for Dropout Rate: {dropout_rate}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"Loss_Curves_Dropout_{dropout_rate:.2f}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot for dropout rate {dropout_rate}: {e}")
        plt.close()

try:
    metrics_train = experiment_data["hyperparam_tuning_dropout"][
        "peer_review_feedback"
    ]["metrics"]["train"]
    plt.figure()
    plt.plot(metrics_train, label="Training Metric", marker="o")
    plt.title("Training Metric Over Dropout Rates")
    plt.xlabel("Dropout Rate Index")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "Training_Metric_Over_Dropout_Rates.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training Metric plot: {e}")
    plt.close()
