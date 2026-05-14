import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training losses for each dataset dimension
for feature_dim in experiment_data["input_feature_variability_analysis"].keys():
    try:
        plt.figure()
        losses = experiment_data["input_feature_variability_analysis"][feature_dim][
            "losses"
        ]["train"]
        plt.plot(range(1, len(losses) + 1), losses, marker="o")
        plt.title(f"Training Loss for {feature_dim}\nLoss Plot")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"{feature_dim}_training_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot for {feature_dim}: {e}")
        plt.close()

# Plot metrics for each dataset dimension
for feature_dim in experiment_data["input_feature_variability_analysis"].keys():
    try:
        plt.figure()
        metrics = experiment_data["input_feature_variability_analysis"][feature_dim][
            "metrics"
        ]["train"]
        plt.plot(range(1, len(metrics) + 1), metrics, marker="o", color="orange")
        plt.title(f"Metrics for {feature_dim}\nMetric Plot")
        plt.xlabel("Epoch")
        plt.ylabel("UES")
        plt.grid()
        plt.savefig(os.path.join(working_dir, f"{feature_dim}_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {feature_dim}: {e}")
        plt.close()
