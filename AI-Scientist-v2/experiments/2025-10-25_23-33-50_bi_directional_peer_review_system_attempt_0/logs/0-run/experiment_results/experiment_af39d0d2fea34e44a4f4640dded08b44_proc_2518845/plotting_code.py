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
    plt.figure()
    plt.plot(experiment_data["both_features"]["losses"]["train"], label="Train Loss")
    plt.plot(experiment_data["both_features"]["losses"]["val"], label="Validation Loss")
    plt.title("Training and Validation Loss for Both Features")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "both_features_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(experiment_data["both_features"]["metrics"]["train"], label="Train RQI")
    plt.plot(experiment_data["both_features"]["metrics"]["val"], label="Validation RQI")
    plt.title("Training and Validation RQI for Both Features")
    plt.xlabel("Epochs")
    plt.ylabel("RQI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "both_features_rqi_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating RQI plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["author_rating_only"]["losses"]["train"], label="Train Loss"
    )
    plt.plot(
        experiment_data["author_rating_only"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Loss for Author Rating Only")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "author_rating_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating author rating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["review_score_only"]["losses"]["train"], label="Train Loss"
    )
    plt.plot(
        experiment_data["review_score_only"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Loss for Review Score Only")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "review_score_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating review score loss plot: {e}")
    plt.close()
