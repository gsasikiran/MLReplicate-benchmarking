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

for hidden_units in experiment_data["hyperparam_tuning"]["hidden_units"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                "losses"
            ]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                "losses"
            ]["val"],
            label="Validation Loss",
        )
        plt.title(f"Training vs Validation Loss (Hidden Units: {hidden_units})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"hidden_units_{hidden_units}_losses.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for hidden units {hidden_units}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(
            experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                "ground_truth"
            ],
            experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                "predictions"
            ],
        )
        plt.plot([0, 1], [0, 1], "r--")  # Reference line
        plt.title(f"Ground Truth vs Predictions (Hidden Units: {hidden_units})")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(
            os.path.join(working_dir, f"hidden_units_{hidden_units}_predictions.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating prediction plot for hidden units {hidden_units}: {e}")
        plt.close()
