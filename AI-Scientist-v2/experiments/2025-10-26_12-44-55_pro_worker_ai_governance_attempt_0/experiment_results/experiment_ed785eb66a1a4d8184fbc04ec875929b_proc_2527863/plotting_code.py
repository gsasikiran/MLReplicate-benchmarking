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

# Plot training and validation losses for AG News
try:
    losses_ag_news = experiment_data["pwii_analysis"]["ag_news"]["losses"]
    epochs_ag_news = range(len(losses_ag_news["train"]))

    plt.figure()
    plt.plot(epochs_ag_news, losses_ag_news["train"], label="Training Loss")
    plt.plot(epochs_ag_news, losses_ag_news["val"], label="Validation Loss")
    plt.title("AG News Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "ag_news_training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AG News loss plot: {e}")
    plt.close()

# Plot training and validation losses for Wikitext
try:
    losses_wikitext = experiment_data["pwii_analysis"]["wikitext"]["losses"]
    epochs_wikitext = range(len(losses_wikitext["train"]))

    plt.figure()
    plt.plot(epochs_wikitext, losses_wikitext["train"], label="Training Loss")
    plt.plot(epochs_wikitext, losses_wikitext["val"], label="Validation Loss")
    plt.title("Wikitext Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "wikitext_training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Wikitext loss plot: {e}")
    plt.close()

# Plot training and validation losses for IMDB
try:
    losses_imdb = experiment_data["pwii_analysis"]["imdb"]["losses"]
    epochs_imdb = range(len(losses_imdb["train"]))

    plt.figure()
    plt.plot(epochs_imdb, losses_imdb["train"], label="Training Loss")
    plt.plot(epochs_imdb, losses_imdb["val"], label="Validation Loss")
    plt.title("IMDB Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "imdb_training_validation_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating IMDB loss plot: {e}")
    plt.close()
