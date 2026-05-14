import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_81da27a3543440a1ac1463effc1c4522_proc_2512891/experiment_data.npy",
    "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_44f20d31049741579473df2cc0443e11_proc_2512892/experiment_data.npy",
    "experiments/2025-10-25_18-43-20_bi_directional_peer_review_system_attempt_0/logs/0-run/experiment_results/experiment_fbd715ae4b084f77b79f6b2ea10eedf9_proc_2512891/experiment_data.npy",
]

all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data: {e}")

# Assuming we have loaded the experiment data
try:
    loss_train = np.array(
        [exp["RQI_experiment"]["losses"]["train"] for exp in all_experiment_data]
    )
    loss_val = np.array(
        [exp["RQI_experiment"]["losses"]["val"] for exp in all_experiment_data]
    )

    mean_loss_train = np.mean(loss_train, axis=0)
    mean_loss_val = np.mean(loss_val, axis=0)
    sem_loss_train = np.std(loss_train, axis=0) / np.sqrt(loss_train.shape[0])
    sem_loss_val = np.std(loss_val, axis=0) / np.sqrt(loss_val.shape[0])

    plt.figure()
    plt.plot(mean_loss_train, label="Mean Training Loss")
    plt.fill_between(
        range(len(mean_loss_train)),
        mean_loss_train - sem_loss_train,
        mean_loss_train + sem_loss_train,
        alpha=0.2,
    )
    plt.plot(mean_loss_val, label="Mean Validation Loss")
    plt.fill_between(
        range(len(mean_loss_val)),
        mean_loss_val - sem_loss_val,
        mean_loss_val + sem_loss_val,
        alpha=0.2,
    )
    plt.title("Mean Loss Over Epochs with SE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mean_losses_over_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean loss plot: {e}")
    plt.close()
