import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()

    # Plot training losses
    plt.figure()
    batch_sizes = list(experiment_data["hyperparam_tuning_batch_size"].keys())
    for batch_size in batch_sizes:
        losses = experiment_data["hyperparam_tuning_batch_size"][batch_size]["losses"][
            "train"
        ]
        plt.plot(losses, label=f"Batch Size {batch_size}")
    plt.title("Training Loss per Batch Size")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss_vs_batch_size.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Plot UES metric
    plt.figure()
    metrics = [
        experiment_data["hyperparam_tuning_batch_size"][batch_size]["metrics"]["train"]
        for batch_size in batch_sizes
    ]
    for batch_size, metric in zip(batch_sizes, metrics):
        plt.plot(metric, label=f"Batch Size {batch_size}")
    plt.title("UES Metric per Batch Size")
    plt.xlabel("Epochs")
    plt.ylabel("UES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ues_metric_vs_batch_size.png"))
    plt.close()
except Exception as e:
    print(f"Error creating UES metric plot: {e}")
    plt.close()

try:
    # Plot predictions if available
    predictions = experiment_data["hyperparam_tuning_batch_size"][batch_sizes[0]][
        "predictions"
    ]
    ground_truth = experiment_data["hyperparam_tuning_batch_size"][batch_sizes[0]][
        "ground_truth"
    ]
    plt.figure()
    plt.scatter(range(len(predictions)), predictions, label="Predictions", color="r")
    plt.scatter(range(len(ground_truth)), ground_truth, label="Ground Truth", color="b")
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
