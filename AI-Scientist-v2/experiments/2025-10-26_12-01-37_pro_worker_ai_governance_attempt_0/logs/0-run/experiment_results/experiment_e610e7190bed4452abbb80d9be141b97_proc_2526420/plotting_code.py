import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    dropout_rates = [0.1, 0.3, 0.5]

    for dropout_rate in dropout_rates:
        plt.figure()
        plt.plot(
            experiment_data["dropout_regularization"]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["dropout_regularization"]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Training and Validation Loss (Dropout {dropout_rate})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"training_validation_loss_dropout_{dropout_rate}.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating loss plots: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        dropout_rates,
        experiment_data["dropout_regularization"]["metrics"]["val"],
        marker="o",
    )
    plt.title("Economic Impact Score (EIS) by Dropout Rate")
    plt.xlabel("Dropout Rate")
    plt.ylabel("EIS")
    plt.xticks(dropout_rates)
    plt.savefig(os.path.join(working_dir, "EIS_by_dropout_rate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating EIS plot: {e}")
    plt.close()
