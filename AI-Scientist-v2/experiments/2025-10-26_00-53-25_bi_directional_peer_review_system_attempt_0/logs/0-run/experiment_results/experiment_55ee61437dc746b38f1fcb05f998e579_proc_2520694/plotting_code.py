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

# Plot for Metrics
try:
    metrics = experiment_data["feature_influence_ablation"]["full"]["metrics"]
    plt.figure()
    plt.plot(metrics["train"], label="Train RQI")
    plt.plot(metrics["val"], label="Validation RQI")
    plt.title("Reviewer Quality Index (RQI) - Full Features")
    plt.xlabel("Epochs")
    plt.ylabel("RQI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "full_features_rqi_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for RQI with full features: {e}")
    plt.close()

try:
    losses = experiment_data["feature_influence_ablation"]["full"]["losses"]
    plt.figure()
    plt.plot(losses["train"], label="Train Loss")
    plt.plot(losses["val"], label="Validation Loss")
    plt.title("Losses - Full Features")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "full_features_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for Loss with full features: {e}")
    plt.close()

# Repeat plotting for other feature ablations
for feature in ["no_clarity", "no_depth", "no_relevance"]:
    try:
        metrics = experiment_data["feature_influence_ablation"][feature]["metrics"]
        plt.figure()
        plt.plot(metrics["train"], label="Train RQI")
        plt.plot(metrics["val"], label="Validation RQI")
        plt.title(f'RQI with {feature.replace("_", " ").title()}')
        plt.xlabel("Epochs")
        plt.ylabel("RQI")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{feature}_rqi_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for RQI with {feature}: {e}")
        plt.close()

    try:
        losses = experiment_data["feature_influence_ablation"][feature]["losses"]
        plt.figure()
        plt.plot(losses["train"], label="Train Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title(f'Losses with {feature.replace("_", " ").title()}')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{feature}_loss_plot.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for Loss with {feature}: {e}")
        plt.close()
