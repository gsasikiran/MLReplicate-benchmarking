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

for difficulty in ["easy", "medium", "hard"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][difficulty]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["multi_dataset_evaluation"][difficulty]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"{difficulty.capitalize()} Sudoku Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{difficulty}_sudoku_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {difficulty} difficulty: {e}")
        plt.close()
