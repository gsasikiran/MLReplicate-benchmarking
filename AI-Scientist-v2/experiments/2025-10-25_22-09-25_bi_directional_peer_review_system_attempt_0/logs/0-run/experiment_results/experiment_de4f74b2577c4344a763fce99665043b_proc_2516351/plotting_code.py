import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plotting training loss
try:
    plt.figure()
    epochs = range(
        1,
        len(
            experiment_data["learning_rate_tuning"]["synthetic_data"]["losses"]["train"]
        )
        + 1,
    )
    plt.plot(
        epochs,
        experiment_data["learning_rate_tuning"]["synthetic_data"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plotting RQS
try:
    plt.figure()
    rqs = experiment_data["learning_rate_tuning"]["synthetic_data"]["metrics"]["train"]
    plt.plot(epochs, rqs, label="RQS", color="orange")
    plt.title("RQS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("RQS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "RQS_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQS plot: {e}")
    plt.close()
