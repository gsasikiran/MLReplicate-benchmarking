import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    activation_analysis = experiment_data["activation_function_analysis"][
        "synthetic_dataset"
    ]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training loss
try:
    plt.figure()
    plt.plot(activation_analysis["losses"]["train"], label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training metrics
try:
    plt.figure()
    plt.plot(
        activation_analysis["metrics"]["train"], label="Collaborative Interaction Score"
    )
    plt.title("Training Metrics (CIS) over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("CIS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training metrics plot: {e}")
    plt.close()

# Plot generated samples (if applicable, for simplicity we assume there are samples)
try:
    plt.figure()
    samples = activation_analysis["predictions"][:5]  # Placeholder logic
    plt.plot(samples, label="Generated Samples")
    plt.title("Generated Samples from Model")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_generated_samples.png"))
    plt.close()
except Exception as e:
    print(f"Error creating generated samples plot: {e}")
    plt.close()
