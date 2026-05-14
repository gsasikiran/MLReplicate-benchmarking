import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training losses for Sentiment Analysis
try:
    plt.figure()
    plt.plot(
        experiment_data["multi_dataset_study"]["sentiment_analysis"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Sentiment Analysis Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "sentiment_analysis_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Sentiment Analysis training loss plot: {e}")
    plt.close()

# Plot training losses for Question Answering
try:
    plt.figure()
    plt.plot(
        experiment_data["multi_dataset_study"]["question_answering"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Question Answering Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "question_answering_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Question Answering training loss plot: {e}")
    plt.close()

# Plot training losses for Text Summarization
try:
    plt.figure()
    plt.plot(
        experiment_data["multi_dataset_study"]["text_summarization"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Text Summarization Training Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "text_summarization_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Text Summarization training loss plot: {e}")
    plt.close()
