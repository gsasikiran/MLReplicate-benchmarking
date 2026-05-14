import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    plt.plot(
        experiment_data["regularization"]["without_l2"]["losses"]["train"],
        label="Training Loss (No L2)",
    )
    plt.plot(
        experiment_data["regularization"]["without_l2"]["losses"]["val"],
        label="Validation Loss (No L2)",
    )
    plt.title("Losses without L2 Regularization")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "losses_without_l2.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for losses without L2: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["regularization"]["with_l2"]["losses"]["train"],
        label="Training Loss (With L2)",
    )
    plt.plot(
        experiment_data["regularization"]["with_l2"]["losses"]["val"],
        label="Validation Loss (With L2)",
    )
    plt.title("Losses with L2 Regularization")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "losses_with_l2.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for losses with L2: {e}")
    plt.close()
