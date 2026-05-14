"""
Final Aggregator Script for Pro-Worker AI Governance Experiments

This script aggregates results from baseline, research, and various ablation studies,
loading numerical results from .npy files and creating publication-ready figures
stored in the "figures/" directory.

Each figure is generated within a try-except block so that one failure does not affect
the rest of the plots. Plots use enlarged fonts, professional styling (without top/right spines),
and are saved at 300 dpi.

The figures include:
  1. Baseline/Research: Training vs Validation Loss for different hidden unit configurations.
  2. Baseline/Research: Scatter plots of Predictions vs Ground Truth for different hidden units.
  3. Dropout Regularization: Aggregated loss curves for different dropout rates.
  4. Dropout Regularization: Economic Impact Score (EIS) vs Dropout Rate.
  5. Model Architecture Variation Ablation: Loss curves for single vs two hidden layer models.
  6. Model Architecture Variation Ablation: Validation Metric (EIS) curves for each architecture.
  7. Input Feature Correlation Ablation (Main): Aggregated loss curves across datasets.
  8. Input Feature Correlation Ablation (Main): Aggregated Predictions vs Ground Truth.
  9. Appendix (Input Feature Correlation): Detailed figures for dataset_1 and dataset_3.
 10. Overall Comparison: Summary plot comparing key final metrics across experiments.
  
Data is loaded exactly from the .npy file paths specified in the summaries.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Create output directories
os.makedirs("figures", exist_ok=True)

# Set global font size for readability.
plt.rcParams.update({'font.size': 14})

# Function to remove top and right spines for a given axis.
def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

###############################################
# 1. Baseline/Research: Training & Validation Loss vs Hidden Units (Aggregated)
###############################################
try:
    baseline_path = "experiment_results/experiment_276439dce56242ad9a2f07c45bbb77de_proc_2525845/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    hidden_units_data = baseline_data["hyperparam_tuning"]["hidden_units"]
    # Sort by hidden unit values for consistent ordering
    hidden_units_sorted = sorted(hidden_units_data.keys(), key=lambda x: int(x))
    
    fig, axs = plt.subplots(1, len(hidden_units_sorted), figsize=(6*len(hidden_units_sorted), 5), dpi=300)
    if len(hidden_units_sorted) == 1:
        axs = [axs]
    for ax, hu in zip(axs, hidden_units_sorted):
        data = hidden_units_data[hu]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs = range(1, len(train_losses)+1)
        ax.plot(epochs, train_losses, label="Train Loss", marker='o')
        ax.plot(epochs, val_losses, label="Validation Loss", marker='o')
        ax.set_title(f"Hidden Units {hu}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        style_axis(ax)
    fig.suptitle("Baseline/Research: Training vs Validation Loss", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join("figures", "baseline_hidden_units_losses.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Baseline Loss Plot:", e)
    
###############################################
# 2. Baseline/Research: Predictions vs Ground Truth (Aggregated)
###############################################
try:
    fig, axs = plt.subplots(1, len(hidden_units_sorted), figsize=(6*len(hidden_units_sorted), 5), dpi=300)
    if len(hidden_units_sorted) == 1:
        axs = [axs]
    for ax, hu in zip(axs, hidden_units_sorted):
        data = hidden_units_data[hu]
        predictions = data["predictions"]
        ground_truth = data["ground_truth"]
        ax.scatter(ground_truth, predictions, label="Predictions", alpha=0.7)
        # Plot the diagonal line for perfect prediction.
        lims = [min(min(ground_truth), min(predictions)), max(max(ground_truth), max(predictions))]
        ax.plot(lims, lims, "r--", label="Ideal")
        ax.set_title(f"Hidden Units {hu}")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.legend()
        style_axis(ax)
    fig.suptitle("Baseline/Research: Predictions vs Ground Truth", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join("figures", "baseline_hidden_units_predictions.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Baseline Predictions Plot:", e)

###############################################
# 3. Dropout Regularization: Loss Curves for Different Dropout Rates (Aggregated)
###############################################
try:
    dropout_path = "experiment_results/experiment_e610e7190bed4452abbb80d9be141b97_proc_2526420/experiment_data.npy"
    dropout_data = np.load(dropout_path, allow_pickle=True).item()
    dropout_dict = dropout_data["dropout_regularization"]["losses"]
    dropout_rates = [0.1, 0.3, 0.5]
    
    fig, axs = plt.subplots(1, len(dropout_rates), figsize=(6*len(dropout_rates), 5), dpi=300)
    if len(dropout_rates) == 1:
        axs = [axs]
    for ax, rate in zip(axs, dropout_rates):
        # Attempt to get loss curves per dropout rate; keys might be stored as strings.
        key = str(rate)
        if key in dropout_dict:
            losses_data = dropout_dict[key]
        else:
            # Fallback: use the whole dict if not split by rate.
            losses_data = dropout_dict
        train_losses = losses_data["train"]
        val_losses = losses_data["val"]
        epochs = range(1, len(train_losses)+1)
        ax.plot(epochs, train_losses, label="Training Loss", marker='o')
        ax.plot(epochs, val_losses, label="Validation Loss", marker='o')
        ax.set_title(f"Dropout {rate}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        style_axis(ax)
    fig.suptitle("Dropout Regularization: Loss Curves", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join("figures", "dropout_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Dropout Loss Curves Plot:", e)

###############################################
# 4. Dropout Regularization: EIS vs Dropout Rate
###############################################
try:
    # Expecting a list of EIS values corresponding to dropout rates [0.1, 0.3, 0.5]
    eis_values = dropout_data["dropout_regularization"]["metrics"]["val"]
    fig, ax = plt.subplots(figsize=(6,5), dpi=300)
    ax.plot(dropout_rates, eis_values, marker="o", linestyle="-", label="EIS")
    ax.set_title("Economic Impact Score (EIS) vs Dropout Rate")
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("EIS")
    ax.set_xticks(dropout_rates)
    ax.legend()
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "EIS_by_dropout_rate.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Dropout EIS Plot:", e)

###############################################
# 5. Model Architecture Variation Ablation: Loss Curves (Single vs Two Hidden Layers)
###############################################
try:
    arch_path = "experiment_results/experiment_db4772265f784f7785da114523cc7736_proc_2526420/experiment_data.npy"
    arch_data = np.load(arch_path, allow_pickle=True).item()
    architectures = arch_data["model_architecture_variation"]
    arch_keys = list(architectures.keys())  # e.g., "single_hidden_layer", "two_hidden_layers"
    
    fig, axs = plt.subplots(1, len(arch_keys), figsize=(7*len(arch_keys), 5), dpi=300)
    if len(arch_keys) == 1:
        axs = [axs]
    for ax, key in zip(axs, arch_keys):
        loss_train = architectures[key]["losses"]["train"]
        loss_val = architectures[key]["losses"]["val"]
        epochs = range(1, len(loss_train)+1)
        ax.plot(epochs, loss_train, label="Training Loss", marker='o')
        ax.plot(epochs, loss_val, label="Validation Loss", marker='o')
        title = key.replace("_", " ").title()
        ax.set_title(f"{title} Loss Curves")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        style_axis(ax)
    fig.suptitle("Model Architecture Variation: Loss Curves", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join("figures", "model_architecture_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Model Architecture Loss Curves:", e)

###############################################
# 6. Model Architecture Variation Ablation: Validation Metric (EIS) Curves
###############################################
try:
    fig, axs = plt.subplots(1, len(arch_keys), figsize=(7*len(arch_keys), 5), dpi=300)
    if len(arch_keys) == 1:
        axs = [axs]
    for ax, key in zip(axs, arch_keys):
        metrics_val = architectures[key]["metrics"]["val"]
        epochs = range(1, len(metrics_val)+1)
        ax.plot(epochs, metrics_val, label="Validation EIS", marker='o')
        title = key.replace("_", " ").title()
        ax.set_title(f"{title} Validation Metric")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("EIS")
        ax.legend()
        style_axis(ax)
    fig.suptitle("Model Architecture Variation: Validation Metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join("figures", "model_architecture_validation_metrics.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Model Architecture Validation Metrics Plot:", e)

###############################################
# 7. Input Feature Correlation Ablation (Main): Loss Curves Aggregated across Datasets
###############################################
try:
    input_corr_path = "experiment_results/experiment_11ba1071667848ea86c99921d100d363_proc_2526419/experiment_data.npy"
    input_corr_data = np.load(input_corr_path, allow_pickle=True).item()
    input_corr = input_corr_data["input_feature_correlation_ablation"]
    datasets = list(input_corr.keys())
    
    fig, axs = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), dpi=300)
    if len(datasets) == 1:
        axs = [axs]
    for ax, dset in zip(axs, datasets):
        losses = input_corr[dset]["losses"]
        train_losses = losses["train"]
        val_losses = losses["val"]
        epochs = range(1, len(train_losses)+1)
        ax.plot(epochs, train_losses, label="Train Loss", marker='o')
        ax.plot(epochs, val_losses, label="Validation Loss", marker='o')
        ax.set_title(f"{dset.replace('_', ' ').title()} Loss Curves")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        style_axis(ax)
    fig.suptitle("Input Feature Correlation: Loss Curves", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join("figures", "input_feature_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Input Feature Loss Curves Plot:", e)

###############################################
# 8. Input Feature Correlation Ablation (Main): Predictions vs Ground Truth (Aggregated)
###############################################
try:
    fig, axs = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), dpi=300)
    if len(datasets) == 1:
        axs = [axs]
    for ax, dset in zip(axs, datasets):
        pred = input_corr[dset]["predictions"]
        gt = input_corr[dset]["ground_truth"]
        ax.scatter(gt, pred, label="Predictions", alpha=0.6)
        lims = [min(min(gt), min(pred)), max(max(gt), max(pred))]
        ax.plot(lims, lims, "r--", label="Ideal")
        ax.set_title(f"{dset.replace('_', ' ').title()} Predictions")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.legend()
        style_axis(ax)
    fig.suptitle("Input Feature Correlation: Predictions vs Ground Truth", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join("figures", "input_feature_predictions.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Input Feature Predictions Plot:", e)

###############################################
# 9. Appendix: Detailed Figures for Selected Datasets (dataset_1 and dataset_3)
###############################################
for dset in ["dataset_1", "dataset_3"]:
    try:
        # Create a combined figure with two subplots: Loss Curves and Predictions vs Ground Truth
        if dset in input_corr:
            losses = input_corr[dset]["losses"]
            pred = input_corr[dset]["predictions"]
            gt = input_corr[dset]["ground_truth"]
            epochs = range(1, len(losses["train"])+1)
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
            # Subplot 1: Loss Curves
            axs[0].plot(epochs, losses["train"], label="Train Loss", marker='o')
            axs[0].plot(epochs, losses["val"], label="Val Loss", marker='o')
            axs[0].set_title(f"{dset.replace('_',' ').title()} Loss Curves (Appendix)")
            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Loss")
            axs[0].legend()
            style_axis(axs[0])
            
            # Subplot 2: Predictions vs Ground Truth
            axs[1].scatter(gt, pred, label="Predictions", alpha=0.6)
            lims = [min(min(gt), min(pred)), max(max(gt), max(pred))]
            axs[1].plot(lims, lims, "r--", label="Ideal")
            axs[1].set_title(f"{dset.replace('_',' ').title()} Predictions (Appendix)")
            axs[1].set_xlabel("Ground Truth")
            axs[1].set_ylabel("Predictions")
            axs[1].legend()
            style_axis(axs[1])
            
            fig.suptitle(f"Appendix: Detailed Analysis for {dset.replace('_',' ').title()}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fname = f"appendix_{dset}.png"
            fig.savefig(os.path.join("figures", fname))
            plt.close(fig)
    except Exception as e:
        print(f"Error in Appendix figure for {dset}:", e)

###############################################
# 10. Overall Comparison: Summary Plot Across Experiments
###############################################
try:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # (a) Baseline: Final Validation Loss vs Hidden Units
    hu_vals = []
    final_losses = []
    for hu in hidden_units_sorted:
        hu_vals.append(int(hu))
        final_losses.append(hidden_units_data[hu]["losses"]["val"][-1])
    axs[0].scatter(hu_vals, final_losses, color='b', label="Final Val Loss")
    axs[0].plot(hu_vals, final_losses, linestyle="--")
    axs[0].set_title("Baseline: Hidden Units vs Final Val Loss")
    axs[0].set_xlabel("Hidden Units")
    axs[0].set_ylabel("Final Validation Loss")
    axs[0].legend()
    style_axis(axs[0])
    
    # (b) Dropout: EIS vs Dropout Rate
    axs[1].plot(dropout_rates, eis_values, marker='o', linestyle="-", color='g', label="EIS")
    axs[1].set_title("Dropout: EIS vs Dropout Rate")
    axs[1].set_xlabel("Dropout Rate")
    axs[1].set_ylabel("EIS")
    axs[1].set_xticks(dropout_rates)
    axs[1].legend()
    style_axis(axs[1])
    
    # (c) Architecture: Final Validation Metric for each Architecture
    arch_names = []
    final_metrics = []
    for key in arch_keys:
        arch_names.append(key.replace("_", " ").title())
        final_metrics.append(architectures[key]["metrics"]["val"][-1])
    axs[2].bar(arch_names, final_metrics, color='orange', label="Final EIS")
    axs[2].set_title("Architecture: Final Validation EIS")
    axs[2].set_xlabel("Model Architecture")
    axs[2].set_ylabel("Final Validation EIS")
    axs[2].legend()
    style_axis(axs[2])
    
    fig.suptitle("Overall Comparison of Experimental Configurations", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join("figures", "overall_comparison.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Overall Comparison Plot:", e)

print("Figure generation complete. All final plots are saved in the 'figures/' directory.")