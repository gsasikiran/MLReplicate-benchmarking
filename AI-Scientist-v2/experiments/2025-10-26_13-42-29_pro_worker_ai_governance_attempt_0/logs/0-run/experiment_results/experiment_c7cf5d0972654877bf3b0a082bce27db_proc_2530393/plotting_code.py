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

try:
    plt.figure()
    plt.plot(
        experiment_data["input_feature_importance"]["all_features"]["losses"]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["input_feature_importance"]["all_features"]["losses"]["val"],
        label="Validation Loss",
    )
    plt.title("Model Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_features_training_validation_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for all features: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["input_feature_importance"]["omit_job_satisfaction"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["input_feature_importance"]["omit_job_satisfaction"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Omitting Job Satisfaction")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "omit_job_satisfaction_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for omitting job satisfaction: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["input_feature_importance"]["omit_job_security"]["losses"][
            "train"
        ],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["input_feature_importance"]["omit_job_security"]["losses"][
            "val"
        ],
        label="Validation Loss",
    )
    plt.title("Omitting Job Security")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "omit_job_security_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for omitting job security: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["input_feature_importance"]["omit_retraining_opportunities"][
            "losses"
        ]["train"],
        label="Train Loss",
    )
    plt.plot(
        experiment_data["input_feature_importance"]["omit_retraining_opportunities"][
            "losses"
        ]["val"],
        label="Validation Loss",
    )
    plt.title("Omitting Retraining Opportunities")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "omit_retraining_opportunities_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for omitting retraining opportunities: {e}")
    plt.close()
