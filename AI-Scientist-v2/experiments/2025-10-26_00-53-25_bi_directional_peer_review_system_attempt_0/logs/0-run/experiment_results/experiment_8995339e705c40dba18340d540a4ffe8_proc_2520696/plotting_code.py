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

try:
    epochs = range(
        1,
        len(experiment_data["feature_influence_ablation"]["full"]["metrics"]["train"])
        + 1,
    )

    plt.figure()
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["full"]["metrics"]["train"],
        label="Train",
    )
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["full"]["metrics"]["val"],
        label="Validation",
    )
    plt.title("Metrics: Full Feature Set")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "metrics_full_feature_set.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["no_clarity"]["metrics"]["train"],
        label="Train: No Clarity",
    )
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["no_clarity"]["metrics"]["val"],
        label="Validation: No Clarity",
    )
    plt.title("Metrics: No Clarity Feature")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "metrics_no_clarity.png"))
    plt.close()
except Exception as e:
    print(f"Error creating No Clarity metrics plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["no_depth"]["metrics"]["train"],
        label="Train: No Depth",
    )
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["no_depth"]["metrics"]["val"],
        label="Validation: No Depth",
    )
    plt.title("Metrics: No Depth Feature")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "metrics_no_depth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating No Depth metrics plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["no_relevance"]["metrics"][
            "train"
        ],
        label="Train: No Relevance",
    )
    plt.plot(
        epochs,
        experiment_data["feature_influence_ablation"]["no_relevance"]["metrics"]["val"],
        label="Validation: No Relevance",
    )
    plt.title("Metrics: No Relevance Feature")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "metrics_no_relevance.png"))
    plt.close()
except Exception as e:
    print(f"Error creating No Relevance metrics plot: {e}")
    plt.close()
