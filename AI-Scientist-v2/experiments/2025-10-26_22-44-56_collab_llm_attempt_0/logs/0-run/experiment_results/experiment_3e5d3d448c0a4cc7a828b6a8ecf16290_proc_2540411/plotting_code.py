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

for input_size in experiment_data["multi_resolution_input_analysis"]:
    try:
        loss_data = experiment_data["multi_resolution_input_analysis"][input_size][
            "losses"
        ]["train"]
        plt.figure()
        plt.plot(range(1, len(loss_data) + 1), loss_data, marker="o")
        plt.title(f"Training Loss for {input_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"{input_size}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {input_size}: {e}")
        plt.close()

    try:
        metric_data = experiment_data["multi_resolution_input_analysis"][input_size][
            "metrics"
        ]["train"]
        plt.figure()
        plt.plot(range(1, len(metric_data) + 1), metric_data, marker="o")
        plt.title(f"User Engagement Score for {input_size}")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"{input_size}_ues.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating UES plot for {input_size}: {e}")
        plt.close()
