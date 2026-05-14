import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_paths = [
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_d76124a4df8f48518ec851db41cc054f_proc_2540412/experiment_data.npy",
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_5a16dd871941475392920fcec54b11d9_proc_2540411/experiment_data.npy",
        "experiments/2025-10-26_22-44-56_collab_llm_attempt_0/logs/0-run/experiment_results/experiment_7fb07d7761ed4a2883a6e40a2364569b_proc_2540414/experiment_data.npy",
    ]
    all_experiment_data = []
    for data_path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), data_path), allow_pickle=True
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

for structure in all_experiment_data[0]["prompt_structure_variation"]:
    losses = []
    ues_metrics = []

    for experiment_data in all_experiment_data:
        losses.append(
            experiment_data["prompt_structure_variation"][structure]["losses"]["train"]
        )
        ues_metrics.append(
            experiment_data["prompt_structure_variation"][structure]["metrics"]["train"]
        )

    # Convert to arrays for mean and std calculations
    losses = np.array(losses)
    ues_metrics = np.array(ues_metrics)

    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0) / np.sqrt(len(losses))
    mean_ues = np.mean(ues_metrics, axis=0)
    std_ues = np.std(ues_metrics, axis=0) / np.sqrt(len(ues_metrics))

    # Plotting Training Loss with error bars
    try:
        plt.figure()
        plt.errorbar(
            range(len(mean_losses)),
            mean_losses,
            yerr=std_losses,
            label="Mean Training Loss",
            capsize=5,
        )
        plt.title(f"Mean Training Loss for {structure}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"Mean_Training_Loss_{structure}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Mean Training Loss plot for {structure}: {e}")
        plt.close()

    # Plotting UES Metric with error bars
    try:
        plt.figure()
        plt.errorbar(
            range(len(mean_ues)),
            mean_ues,
            yerr=std_ues,
            label="Mean UES Metric",
            color="orange",
            capsize=5,
        )
        plt.title(f"Mean UES Metric for {structure}")
        plt.xlabel("Epochs")
        plt.ylabel("UES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"Mean_UES_Metric_{structure}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Mean UES Metric plot for {structure}: {e}")
        plt.close()
