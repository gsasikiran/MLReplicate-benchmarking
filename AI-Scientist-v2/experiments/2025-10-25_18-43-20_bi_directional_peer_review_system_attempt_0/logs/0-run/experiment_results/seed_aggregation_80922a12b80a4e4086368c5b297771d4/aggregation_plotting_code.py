import matplotlib.pyplot as plt
import numpy as np
import os

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    # Load experiment data
    experiment_data_path_list = [
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_273c1c577f7d4d7bbe0d4ffe91b7aa4d_proc_2513037/experiment_data.npy",
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_a91fb9c2eaf74aeb8a93b3a110379cb2_proc_2513038/experiment_data.npy",
        "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_44d5c35955174420b86c3d1927f5ee0e_proc_2513037/experiment_data.npy",
    ]

    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Prepare data for plotting
    num_epochs = len(
        all_experiment_data[0]["hyperparam_tuning_num_hidden_units"][
            "synthetic_dataset"
        ]["losses"]["train"]
    )
    train_losses = np.array(
        [
            exp["hyperparam_tuning_num_hidden_units"]["synthetic_dataset"]["losses"][
                "train"
            ]
            for exp in all_experiment_data
        ]
    )
    val_losses = np.array(
        [
            exp["hyperparam_tuning_num_hidden_units"]["synthetic_dataset"]["losses"][
                "val"
            ]
            for exp in all_experiment_data
        ]
    )
    mean_train_losses = np.mean(train_losses, axis=0)
    mean_val_losses = np.mean(val_losses, axis=0)
    ste_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(all_experiment_data))
    ste_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(all_experiment_data))

    # Plotting
    plt.figure()
    epochs = range(num_epochs)
    plt.plot(epochs, mean_train_losses, label="Mean Training Loss", color="blue")
    plt.fill_between(
        epochs,
        mean_train_losses - ste_train_losses,
        mean_train_losses + ste_train_losses,
        color="blue",
        alpha=0.1,
        label="SE Training Loss",
    )
    plt.plot(epochs, mean_val_losses, label="Mean Validation Loss", color="orange")
    plt.fill_between(
        epochs,
        mean_val_losses - ste_val_losses,
        mean_val_losses + ste_val_losses,
        color="orange",
        alpha=0.1,
        label="SE Validation Loss",
    )

    plt.title("Training and Validation Losses for Synthetic Dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_dataset_mean_training_validation_losses.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean training and validation loss plot: {e}")
    plt.close()
