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

# Plot training and validation losses
for dataset_name in experiment_data["ablation_study"]:
    for hidden_units in experiment_data["ablation_study"][dataset_name][
        "hyperparam_tuning"
    ]["hidden_units"]:
        try:
            plt.figure()
            train_losses = experiment_data["ablation_study"][dataset_name][
                "hyperparam_tuning"
            ]["hidden_units"][hidden_units]["losses"]["train"]
            val_losses = experiment_data["ablation_study"][dataset_name][
                "hyperparam_tuning"
            ]["hidden_units"][hidden_units]["losses"]["val"]
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title(f"{dataset_name}: Training and Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{dataset_name}_train_val_loss_units_{hidden_units}.png",
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating loss plot for {dataset_name} with hidden units {hidden_units}: {e}"
            )
            plt.close()

        try:
            plt.figure()
            predictions = experiment_data["ablation_study"][dataset_name][
                "hyperparam_tuning"
            ]["hidden_units"][hidden_units]["predictions"]
            ground_truth = experiment_data["ablation_study"][dataset_name][
                "hyperparam_tuning"
            ]["hidden_units"][hidden_units]["ground_truth"]
            plt.scatter(ground_truth, predictions)
            plt.title(
                f"{dataset_name}: Predictions vs Ground Truth for Hidden Units {hidden_units}"
            )
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # diagonal line
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"{dataset_name}_predictions_vs_ground_truth_units_{hidden_units}.png",
                )
            )
            plt.close()
        except Exception as e:
            print(
                f"Error creating predictions vs ground truth plot for {dataset_name} with hidden units {hidden_units}: {e}"
            )
            plt.close()
