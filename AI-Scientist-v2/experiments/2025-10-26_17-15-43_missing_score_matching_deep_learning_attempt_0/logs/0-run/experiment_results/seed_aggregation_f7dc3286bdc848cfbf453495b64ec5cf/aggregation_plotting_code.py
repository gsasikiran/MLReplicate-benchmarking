markdown
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_bcaf7a6c070e40a08c0a8d624abb24e3_proc_2534716/experiment_data.npy",
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_bbe975a218424dfe8fe3d1edc5fe3363_proc_2534714/experiment_data.npy",
        "experiments/2025-10-26_17-15-43_missing_score_matching_deep_learning_attempt_0/logs/0-run/experiment_results/experiment_c49802f68024450cb0f9d6f9e200e97c_proc_2534715/experiment_data.npy",
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
    plt.figure()
    for hidden_layer_size, data in all_experiment_data[0][
        "hyperparam_tuning_hidden_layer_size"
    ].items():
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]

        mean_train = np.mean(train_losses, axis=0)
        mean_val = np.mean(val_losses, axis=0)

        se_train = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        se_val = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

        epochs = np.arange(len(mean_train))
        plt.plot(epochs, mean_train, label=f"Mean Train {hidden_layer_size}")
        plt.fill_between(
            epochs, mean_train - se_train, mean_train + se_train, alpha=0.2
        )
        plt.plot(epochs, mean_val, label=f"Mean Val {hidden_layer_size}")
        plt.fill_between(epochs, mean_val - se_val, mean_val + se_val, alpha=0.2)

    plt.title("Mean Training and Validation Loss Curves with Standard Error")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_loss_curves_with_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss curves plot: {e}")
    plt.close()
