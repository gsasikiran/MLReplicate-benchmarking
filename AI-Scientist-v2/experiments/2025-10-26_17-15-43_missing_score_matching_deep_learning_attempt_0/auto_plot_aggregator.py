#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# Use a safe style; if the desired style is not available, fall back to default.
try:
    plt.style.use("seaborn-v0_8")  # use a known valid seaborn style name
except Exception:
    plt.style.use("default")
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12
})

os.makedirs("figures", exist_ok=True)

# Plot 1: Hyperparameter Tuning Loss Curves (Training and Validation side by side)
try:
    hp_path = "experiment_results/experiment_7e4c3fc49de647d5a28009d4430fdeaa_proc_2534399/experiment_data.npy"
    hp_data = np.load(hp_path, allow_pickle=True).item().get("hyperparam_tuning_hidden_layer_size", {})
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    for hidden_size, results in hp_data.items():
        epochs = range(1, len(results["losses"]["train"]) + 1)
        ax[0].plot(epochs, results["losses"]["train"], label=f"Train (Size {hidden_size})")
        ax[1].plot(epochs, results["losses"]["val"], label=f"Validation (Size {hidden_size})")
    ax[0].set_title("Training Loss vs Epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].set_title("Validation Loss vs Epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    fig.suptitle("Hyperparameter Tuning: Loss Curves for Varying Hidden Layer Sizes")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join("figures", "Hyperparameter Tuning Losses.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Plot 1:", e)

# Plot 2: Activation Function Comparison (Selected Configurations)
try:
    act_path = "experiment_results/experiment_c3e083dc2ce34698889cc64e034106f4_proc_2535213/experiment_data.npy"
    act_data = np.load(act_path, allow_pickle=True).item().get("Effect of Activation Functions", {})
    rep_configs = [("ReLU", "128"), ("Sigmoid", "32"), ("Leaky ReLU", "128")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    for idx, (act_name, hidden_size) in enumerate(rep_configs):
        if act_name in act_data and hidden_size in act_data[act_name]:
            subdata = act_data[act_name][hidden_size]
            epochs = range(1, len(subdata["losses"]["train"]) + 1)
            ax[idx].plot(epochs, subdata["losses"]["train"], label="Train Loss")
            ax[idx].plot(epochs, subdata["losses"]["val"], label="Validation Loss")
            ax[idx].set_title(f"{act_name} Activation (Size {hidden_size})")
            ax[idx].set_xlabel("Epochs")
            ax[idx].set_ylabel("Loss")
            ax[idx].legend()
        else:
            ax[idx].text(0.5, 0.5, f"No Data for {act_name} Size {hidden_size}",
                         horizontalalignment="center", verticalalignment="center")
    fig.suptitle("Activation Functions: Selected Loss Curves")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join("figures", "Activation Functions Comparison.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Plot 2:", e)

# Plot 3: Input Data Preprocessing (Mean Imputation vs Data Normalization)
try:
    norm_path = "experiment_results/experiment_e841eaf5e4f149f9b58d8a59c6aa0c3c_proc_2535214/experiment_data.npy"
    norm_data = np.load(norm_path, allow_pickle=True).item().get("input_data_normalization", {})
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    if "mean_imputation" in norm_data:
        mi = norm_data["mean_imputation"]
        epochs = range(1, len(mi["losses"]["train"]) + 1)
        ax[0].plot(epochs, mi["losses"]["train"], label="Train Loss")
        ax[0].plot(epochs, mi["losses"]["val"], label="Validation Loss")
        ax[0].set_title("Mean Imputation Loss Curves")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
    else:
        ax[0].text(0.5, 0.5, "No Data for Mean Imputation",
                   horizontalalignment="center", verticalalignment="center")
    if "data_normalization" in norm_data:
        dn = norm_data["data_normalization"]
        epochs = range(1, len(dn["losses"]["train"]) + 1)
        ax[1].plot(epochs, dn["losses"]["train"], label="Train Loss")
        ax[1].plot(epochs, dn["losses"]["val"], label="Validation Loss")
        ax[1].set_title("Data Normalization Loss Curves")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "No Data for Data Normalization",
                   horizontalalignment="center", verticalalignment="center")
    fig.suptitle("Input Data Preprocessing: Mean Imputation vs Data Normalization")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join("figures", "Input Data Preprocessing.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Plot 3:", e)

# Plot 4: Impact of Batch Normalization on Model Training (With versus Without BN)
try:
    bn_path = "experiment_results/experiment_490b9aebda86436b889cca0947d94aa7_proc_2535215/experiment_data.npy"
    bn_data = np.load(bn_path, allow_pickle=True).item().get("impact_of_batch_normalization", {})
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    if "with_bn" in bn_data:
        with_bn = bn_data["with_bn"]
        epochs = range(1, len(with_bn["losses"]["train"]) + 1)
        ax[0].plot(epochs, with_bn["losses"]["train"], label="Train Loss With BN")
        ax[0].plot(epochs, with_bn["losses"]["val"], label="Validation Loss With BN")
        ax[0].set_title("Loss Curves With Batch Normalization")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
    else:
        ax[0].text(0.5, 0.5, "No Data for With BN",
                   horizontalalignment="center", verticalalignment="center")
    if "without_bn" in bn_data:
        without_bn = bn_data["without_bn"]
        if "epochs" not in locals():
            epochs = range(1, len(without_bn["losses"]["train"]) + 1)
        ax[1].plot(epochs, without_bn["losses"]["train"], label="Train Loss Without BN")
        ax[1].plot(epochs, without_bn["losses"]["val"], label="Validation Loss Without BN")
        ax[1].set_title("Loss Curves Without Batch Normalization")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "No Data for Without BN",
                   horizontalalignment="center", verticalalignment="center")
    fig.suptitle("Impact of Batch Normalization on Model Training")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join("figures", "Batch Normalization Impact.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Plot 4:", e)

# Plot 5: Dropout Regularization Effects on Loss Curves (Sorted Dropout Rates)
try:
    dropout_path = "experiment_results/experiment_e4ba666f2347474892829e20c872132e_proc_2535214/experiment_data.npy"
    dropout_data = np.load(dropout_path, allow_pickle=True).item().get("dropout_regularization_effect", {})
    selected_rates = ["0.5", "0.0", "0.2"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    for idx, rate in enumerate(selected_rates):
        if rate in dropout_data:
            drop_data = dropout_data[rate]
            epochs = range(1, len(drop_data["losses"]["train"]) + 1)
            ax[idx].plot(epochs, drop_data["losses"]["train"], label="Train Loss")
            ax[idx].plot(epochs, drop_data["losses"]["val"], label="Validation Loss")
            ax[idx].set_title(f"Dropout Rate {rate}")
            ax[idx].set_xlabel("Epochs")
            ax[idx].set_ylabel("Loss")
            ax[idx].legend()
        else:
            ax[idx].text(0.5, 0.5, f"No Data for Dropout Rate {rate}",
                         horizontalalignment="center", verticalalignment="center")
    fig.suptitle("Dropout Regularization Effects on Loss Curves")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join("figures", "Dropout Regularization Effects.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Plot 5:", e)

# Plot 6: Impact of Varying Missing Data Patterns on Loss (MCAR, MAR, NMAR)
try:
    missing_path = "experiment_results/experiment_35b6f2cc94ba473c83574c8302d931a8_proc_2535214/experiment_data.npy"
    missing_data = np.load(missing_path, allow_pickle=True).item().get("missing_data_patterns", {})
    patterns = ["mcar", "mar", "nmar"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    for idx, pattern in enumerate(patterns):
        if pattern in missing_data:
            pat_data = missing_data[pattern]
            epochs = range(1, len(pat_data["losses"]["train"]) + 1)
            ax[idx].plot(epochs, pat_data["losses"]["train"], label="Train Loss")
            ax[idx].plot(epochs, pat_data["losses"]["val"], label="Validation Loss")
            ax[idx].set_title(f"{pattern.upper()} Data Loss Curves")
            ax[idx].set_xlabel("Epochs")
            ax[idx].set_ylabel("Loss")
            ax[idx].legend()
        else:
            ax[idx].text(0.5, 0.5, f"No Data for {pattern.upper()}",
                         horizontalalignment="center", verticalalignment="center")
    fig.suptitle("Impact of Varying Missing Data Patterns on Loss")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join("figures", "Missing Data Patterns Impact.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Plot 6:", e)

print("All final plots have been generated and saved in the 'figures' folder.")