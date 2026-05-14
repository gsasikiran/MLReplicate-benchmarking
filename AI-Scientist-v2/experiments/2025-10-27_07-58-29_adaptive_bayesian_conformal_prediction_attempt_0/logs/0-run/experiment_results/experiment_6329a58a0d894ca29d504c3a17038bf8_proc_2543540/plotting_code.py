import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

l2_lambda_values = list(experiment_data["hyperparam_tuning_L2"].keys())

# Plot loss curves for training and validation
try:
    plt.figure()
    for l2_lambda in l2_lambda_values:
        train_losses = experiment_data["hyperparam_tuning_L2"][l2_lambda]["losses"][
            "train"
        ]
        val_losses = experiment_data["hyperparam_tuning_L2"][l2_lambda]["losses"]["val"]
        plt.plot(train_losses, label=f"Train Loss (λ={l2_lambda})")
        plt.plot(val_losses, label=f"Validation Loss (λ={l2_lambda})", linestyle="--")
    plt.title("Training and Validation Losses Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot reliability measures
try:
    reliability_measures = [
        experiment_data["hyperparam_tuning_L2"][l]["metrics"]["val"][0]
        for l in l2_lambda_values
    ]
    plt.figure()
    plt.bar(l2_lambda_values, reliability_measures)
    plt.title("Reliability Measures by L2 Regularization")
    plt.xlabel("L2 Regularization Strength")
    plt.ylabel("Reliability Measure")
    plt.savefig(os.path.join(working_dir, "reliability_measures.png"))
    plt.close()
except Exception as e:
    print(f"Error creating reliability plot: {e}")
    plt.close()
