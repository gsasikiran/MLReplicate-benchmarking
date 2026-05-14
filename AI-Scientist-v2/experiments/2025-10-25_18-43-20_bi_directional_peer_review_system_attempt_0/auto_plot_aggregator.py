#!/usr/bin/env python3
"""
Aggregator script for final scientific paper figures.
This script loads experiment data from multiple npy files and aggregates final plots.
All figures are saved in the "figures" directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Global style settings
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Create final figures directory
os.makedirs("figures", exist_ok=True)

###############################################################################
# Plot 1: Baseline – Synthetic Dataset Training and Validation Losses
###############################################################################
try:
    baseline_path = "experiment_results/experiment_9fdf9a7494424c76b930d64080f48508_proc_2513038/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    # Access synthetic dataset losses from hyperparam tuning node
    losses = baseline_data["hyperparam_tuning_num_hidden_units"]["synthetic_dataset"]["losses"]
    train_losses = losses["train"]
    val_losses = losses["val"]
    epochs = range(len(train_losses))
    
    plt.figure(dpi=300)
    plt.plot(epochs, train_losses, label="Training Loss", lw=2)
    plt.plot(epochs, val_losses, label="Validation Loss", lw=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Baseline: Synthetic Dataset Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Baseline_Synthetic_Losses.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 1 (Baseline Synthetic Losses): {e}")

###############################################################################
# Plot 2: Research – Activation Function Tuning (ReLU, Tanh, LeakyReLU, ELU)
###############################################################################
try:
    research_path = "experiment_results/experiment_9d27417e0c8b48cab3020412a3700e46_proc_2513037/experiment_data.npy"
    research_data = np.load(research_path, allow_pickle=True).item()
    act_tuning = research_data["activation_function_tuning"]
    act_names = list(act_tuning.keys())
    n_funcs = len(act_names)
    
    fig, axes = plt.subplots(1, n_funcs, figsize=(5*n_funcs, 4), dpi=300)
    if n_funcs == 1:
        axes = [axes]
    for ax, act in zip(axes, act_names):
        data = act_tuning[act]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs = range(len(train_losses))
        ax.plot(epochs, train_losses, label="Train Loss", lw=2)
        ax.plot(epochs, val_losses, label="Validation Loss", lw=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"{act} Activation")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Research_Activation_Function_Tuning.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 2 (Research Activation Tuning): {e}")

###############################################################################
# Plot 3: Ablation – Multi-Dataset Performance Loss Curves
###############################################################################
try:
    ablation1_path = "experiment_results/experiment_348f9a377e704de6bccdfc0d3c3fb289_proc_2513756/experiment_data.npy"
    ablation1_data = np.load(ablation1_path, allow_pickle=True).item()
    multi_ds = ablation1_data["multi_dataset_performance"]
    # Expected keys: e.g., "Beta", "Uniform", "Normal"
    dataset_names = list(multi_ds.keys())
    n_datasets = len(dataset_names)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4), dpi=300)
    if n_datasets == 1:
        axes = [axes]
    for ax, ds in zip(axes, dataset_names):
        data = multi_ds[ds]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs = range(len(train_losses))
        ax.plot(epochs, train_losses, label="Train Loss", lw=2)
        ax.plot(epochs, val_losses, label="Validation Loss", lw=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"{ds} Dataset")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Ablation_Multi_Dataset_Performance.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 3 (Multi-Dataset Performance): {e}")

###############################################################################
# Plot 4: Ablation – Training Epochs Impact (10, 50, 100 Epochs)
###############################################################################
try:
    epochs_impact_path = "experiment_results/experiment_6a3b8dd26f6d4b53b471c82e68f6d535_proc_2513756/experiment_data.npy"
    epochs_data = np.load(epochs_impact_path, allow_pickle=True).item()
    epoch_keys = ["10_epochs", "50_epochs", "100_epochs"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    for ax, key in zip(axes, epoch_keys):
        data = epochs_data["training_epochs_impact"][key]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs_range = range(len(train_losses))
        ax.plot(epochs_range, train_losses, label="Train Loss", lw=2)
        ax.plot(epochs_range, val_losses, label="Validation Loss", lw=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"Impact of {key.replace('_', ' ')}")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Training_Epochs_Impact.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 4 (Training Epochs Impact): {e}")

###############################################################################
# Plot 5: Ablation – Learning Rate Sensitivity (Tanh, ELU, LeakyReLU, ReLU)
###############################################################################
try:
    lr_path = "experiment_results/experiment_d5b9b8a6215b43048562cb5eb95dc279_proc_2513756/experiment_data.npy"
    lr_data = np.load(lr_path, allow_pickle=True).item()
    lr_dict = lr_data["learning_rate_sensitivity"]
    act_funcs = list(lr_dict.keys())
    n_funcs = len(act_funcs)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    for ax, act in zip(axes, act_funcs):
        data = lr_dict[act]
        losses = data["losses"]
        train_losses = losses["train"]
        val_losses = losses["val"]
        epochs_range = np.arange(len(train_losses))
        ax.plot(epochs_range, train_losses, label="Train Loss", lw=2)
        ax.plot(epochs_range, val_losses, label="Validation Loss", lw=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"{act} Activation")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Learning_Rate_Sensitivity.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 5 (Learning Rate Sensitivity): {e}")

###############################################################################
# Plot 6: Ablation – Activation Function Variety Comparison
# Aggregated train loss curves for various activation functions
###############################################################################
try:
    variety_path = "experiment_results/experiment_e26b4d141abf4020ae40d5abeff74ef5_proc_2513755/experiment_data.npy"
    variety_data = np.load(variety_path, allow_pickle=True).item()
    act_variety = variety_data["activation_function_variation"]
    act_names = list(act_variety.keys())
    
    plt.figure(figsize=(7,5), dpi=300)
    for act in act_names:
        train_losses = act_variety[act]["losses"]["train"]
        epochs_range = range(len(train_losses))
        plt.plot(epochs_range, train_losses, lw=2, label=act)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Activation Function Variety (Training Loss Comparison)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Activation_Function_Variation.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 6 (Activation Function Variety): {e}")

###############################################################################
# Plot 7: Ablation – Multiple Activation Function Comparison (Layer Combinations)
# Four subplots: Combination 1 and 2 (training & validation)
###############################################################################
try:
    multi_act_path = "experiment_results/experiment_f5782fff0c9c428e90039eb04593ce2e_proc_2513755/experiment_data.npy"
    multi_act_data = np.load(multi_act_path, allow_pickle=True).item()
    multi_comp = multi_act_data["multiple_activation_function_comparison"]
    combinations = ["activation_combination_1", "activation_combination_2"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    for i, comb in enumerate(combinations):
        # Training loss subplot (top row)
        ax = axes[0, i]
        train_losses = multi_comp[comb]["losses"]["train"]
        epochs_range = range(len(train_losses))
        ax.plot(epochs_range, train_losses, lw=2, label="Train Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"{comb.replace('_', ' ').title()}: Training Loss")
        ax.legend()
        ax.grid(True)
        # Validation loss subplot (bottom row)
        ax = axes[1, i]
        val_losses = multi_comp[comb]["losses"]["val"]
        ax.plot(epochs_range, val_losses, lw=2, label="Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"{comb.replace('_', ' ').title()}: Validation Loss")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Multiple_Activation_Function_Comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 7 (Multiple Activation Function Comparison): {e}")

###############################################################################
# Plot 8: Ablation – Noise Sensitivity Ablation Loss Curves
# 2x2 subplots for each activation function (Tanh, ELU, LeakyReLU, ReLU)
###############################################################################
try:
    noise_path = "experiment_results/experiment_51f23baf4ace448c91543899b71e1526_proc_2513756/experiment_data.npy"
    noise_data = np.load(noise_path, allow_pickle=True).item()
    noise_dict = noise_data["noise_sensitivity_ablation"]
    noise_funcs = list(noise_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    for ax, func in zip(axes, noise_funcs):
        data = noise_dict[func]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs_range = range(len(train_losses))
        ax.plot(epochs_range, train_losses, label="Train Loss", lw=2)
        ax.plot(epochs_range, val_losses, label="Validation Loss", lw=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"{func} Activation")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Noise_Sensitivity_Loss.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 8 (Noise Sensitivity Loss Curves): {e}")

###############################################################################
# Plot 9: Ablation – Noise Sensitivity Predictions vs Ground Truth
# 2x2 subplots for each activation function (Tanh, ELU, LeakyReLU, ReLU)
###############################################################################
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    for ax, func in zip(axes, noise_funcs):
        data = noise_dict[func]
        gt = data["ground_truth"]
        preds = data["predictions"]
        ax.scatter(gt, preds, alpha=0.5)
        ax.plot([min(gt), max(gt)], [min(gt), max(gt)], "r--", label="Ideal")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.set_title(f"{func} Activation")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Noise_Sensitivity_Predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 9 (Noise Sensitivity Predictions): {e}")

###############################################################################
# Plot 10: Ablation – Multiple Synthetic Datasets Loss Curves
# 1x3 subplots for datasets 1, 2, 3
###############################################################################
try:
    synth_path = "experiment_results/experiment_cc4ed2fca0e84718afb49c930bc61c99_proc_2513755/experiment_data.npy"
    synth_data = np.load(synth_path, allow_pickle=True).item()
    synth_dict = synth_data["multiple_synthetic_datasets"]
    dataset_keys = sorted(synth_dict.keys())  # e.g., dataset_1, dataset_2, dataset_3
    n_datasets = len(dataset_keys)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4), dpi=300)
    if n_datasets == 1:
        axes = [axes]
    for ax, key in zip(axes, dataset_keys):
        data = synth_dict[key]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs_range = range(len(train_losses))
        ax.plot(epochs_range, train_losses, label="Train Loss", lw=2)
        ax.plot(epochs_range, val_losses, label="Validation Loss", lw=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"{key.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Multiple_Synthetic_Datasets_Loss.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 10 (Multiple Synthetic Datasets Loss): {e}")

###############################################################################
# Plot 11: Ablation – Multiple Synthetic Datasets Predictions vs Ground Truth
# 1x3 scatter subplots for datasets 1, 2, 3
###############################################################################
try:
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 4), dpi=300)
    if n_datasets == 1:
        axes = [axes]
    for ax, key in zip(axes, dataset_keys):
        data = synth_dict[key]
        gt = data["ground_truth"]
        preds = data["predictions"]
        ax.scatter(gt, preds, alpha=0.5)
        ax.plot([min(gt), max(gt)], [min(gt), max(gt)], "r--", label="Ideal")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.set_title(f"{key.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Multiple_Synthetic_Datasets_Predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 11 (Multiple Synthetic Datasets Predictions): {e}")

###############################################################################
# Plot 12: Ablation – Feature Impact Ablation (Activation Function Impact)
# 2x2 subplots for different activation functions in feature impact ablation
###############################################################################
try:
    feat_imp_path = "experiment_results/experiment_1c9ca05ca3584f329f3019dc263625c1_proc_2513756/experiment_data.npy"
    feat_imp_data = np.load(feat_imp_path, allow_pickle=True).item()
    feat_dict = feat_imp_data["feature_impact_ablation"]
    feat_funcs = list(feat_dict.keys())
    n_feat = len(feat_funcs)
    
    # Create a 2x2 plot (if there are 4, else adjust)
    ncols = 2
    nrows = (n_feat + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5*nrows), dpi=300)
    axes = axes.flatten() if n_feat > 1 else [axes]
    for ax, func in zip(axes, feat_funcs):
        data = feat_dict[func]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs_range = range(len(train_losses))
        ax.plot(epochs_range, train_losses, label="Train Loss", lw=2)
        ax.plot(epochs_range, val_losses, label="Validation Loss", lw=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(f"Feature Impact: {func}")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Feature_Impact_Ablation.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 12 (Feature Impact Ablation): {e}")

print("All figures have been generated and saved in the 'figures' directory.")