#!/usr/bin/env python3
"""
Aggregated Final Figures for the Bi-Directional Peer Review System Paper

This script loads experiment data from various .npy files
(as generated from the baseline, research, and ablation experiments)
and produces a final set of aggregated, publication‐ready figures.
All outputs are saved in the "figures/" folder.
Each plotting section is wrapped with try/except so that a failure in one does not block others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Improve global aesthetics for publication-quality figures
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create the figures directory if it does not exist
os.makedirs("figures", exist_ok=True)

# ---------------------------
# 1. BASELINE: RQS Metrics & Losses
# ---------------------------
try:
    # Load baseline experiment data
    baseline_file = "experiment_results/experiment_1dc81dd0e03b4e15803c79266dc63623_proc_2519724/experiment_data.npy"
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    
    # Extract training/validation metrics and losses (assume hyperparam_tuning_batch_size under "RQS")
    train_metric = baseline_data["hyperparam_tuning_batch_size"]["RQS"]["metrics"]["train"]
    val_metric = baseline_data["hyperparam_tuning_batch_size"]["RQS"]["metrics"]["val"]
    train_loss = baseline_data["hyperparam_tuning_batch_size"]["RQS"]["losses"]["train"]
    val_loss = baseline_data["hyperparam_tuning_batch_size"]["RQS"]["losses"]["val"]
    epochs = range(1, len(train_metric) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax1.plot(epochs, train_metric, label="Training Metric", marker="o")
    ax1.plot(epochs, val_metric, label="Validation Metric", marker="o")
    ax1.set_title("RQS: Training vs Validation Metric")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Metric Value")
    ax1.legend()

    ax2.plot(epochs, train_loss, label="Training Loss", marker="o")
    ax2.plot(epochs, val_loss, label="Validation Loss", marker="o")
    ax2.set_title("RQS: Training vs Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join("figures", "baseline_RQS.png"), dpi=300)
    plt.close(fig)
    print("Baseline plot saved.")
except Exception as e:
    print("Error in baseline plot:", e)


# ---------------------------
# 2. RESEARCH: RQS Loss, Metric & Predictions
# ---------------------------
try:
    research_file = "experiment_results/experiment_983582410f4848bd8d511127f321aaed_proc_2519723/experiment_data.npy"
    research_data = np.load(research_file, allow_pickle=True).item()

    # Extract losses, metrics, and prediction info (under hyperparam_tuning_lr -> RQS)
    train_loss = research_data["hyperparam_tuning_lr"]["RQS"]["losses"]["train"]
    val_loss   = research_data["hyperparam_tuning_lr"]["RQS"]["losses"]["val"]
    train_metric = research_data["hyperparam_tuning_lr"]["RQS"]["metrics"]["train"]
    val_metric   = research_data["hyperparam_tuning_lr"]["RQS"]["metrics"]["val"]
    predictions = np.squeeze(np.array(research_data["hyperparam_tuning_lr"]["RQS"]["predictions"]))
    ground_truth = np.squeeze(np.array(research_data["hyperparam_tuning_lr"]["RQS"]["ground_truth"]))
    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    
    # Loss subplot
    axes[0].plot(epochs, train_loss, label="Training Loss", marker="o")
    axes[0].plot(epochs, val_loss, label="Validation Loss", marker="o")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    # Metrics subplot
    axes[1].plot(epochs, train_metric, label="Training Metric", marker="o")
    axes[1].plot(epochs, val_metric, label="Validation Metric", marker="o")
    axes[1].set_title("Training vs Validation Metric")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Metric Value")
    axes[1].legend()
    
    # Predictions vs Ground Truth subplot
    axes[2].scatter(ground_truth, predictions, alpha=0.7)
    axes[2].plot([ground_truth.min(), ground_truth.max()],
                 [ground_truth.min(), ground_truth.max()],
                 "r--", label="Line of equality")
    axes[2].set_title("Predictions vs Ground Truth")
    axes[2].set_xlabel("Ground Truth")
    axes[2].set_ylabel("Predictions")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(os.path.join("figures", "research_RQS.png"), dpi=300)
    plt.close(fig)
    print("Research plot saved.")
except Exception as e:
    print("Error in research plot:", e)


# ---------------------------
# 3. ABLATION: Dataset Variation Ablation
#    Aggregate accuracy and loss curves from a selection of datasets.
# ---------------------------
try:
    ablation_dataset_file = "experiment_results/experiment_34e3e9f20bde427ba0b1d7468d019163_proc_2520694/experiment_data.npy"
    data_var_data = np.load(ablation_dataset_file, allow_pickle=True).item()
    ablation_dict = data_var_data["dataset_variation_ablation"]  # dict of different dataset variations
    datasets = list(ablation_dict.keys())
    
    # Aggregate Accuracy curves
    fig_acc, ax_acc = plt.subplots(figsize=(8,6))
    for d in datasets:
        acc_train = ablation_dict[d]["metrics"]["train"]
        acc_val = ablation_dict[d]["metrics"]["val"]
        ax_acc.plot(range(1, len(acc_train)+1), acc_train, marker="o", label=f"{d} - Train")
        ax_acc.plot(range(1, len(acc_val)+1), acc_val, marker="o", linestyle="--", label=f"{d} - Val")
    ax_acc.set_title("Dataset Variation: Training & Validation Accuracy")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    fig_acc.tight_layout()
    fig_acc.savefig(os.path.join("figures", "ablation_dataset_variation_accuracy.png"), dpi=300)
    plt.close(fig_acc)
    
    # Aggregate Loss curves
    fig_loss, ax_loss = plt.subplots(figsize=(8,6))
    for d in datasets:
        loss_train = ablation_dict[d]["losses"]["train"]
        loss_val = ablation_dict[d]["losses"]["val"]
        ax_loss.plot(range(1, len(loss_train)+1), loss_train, marker="o", label=f"{d} - Train")
        ax_loss.plot(range(1, len(loss_val)+1), loss_val, marker="o", linestyle="--", label=f"{d} - Val")
    ax_loss.set_title("Dataset Variation: Training & Validation Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    fig_loss.tight_layout()
    fig_loss.savefig(os.path.join("figures", "ablation_dataset_variation_loss.png"), dpi=300)
    plt.close(fig_loss)
    print("Dataset Variation Ablation plots saved.")
except Exception as e:
    print("Error in Dataset Variation Ablation plots:", e)


# ---------------------------
# 4. ABLATION: Activation Function Ablation
#    Aggregate accuracy and loss curves for various activation functions.
# ---------------------------
try:
    act_file = "experiment_results/experiment_c6fb7f50328f452f87a600dc01fea347_proc_2520696/experiment_data.npy"
    act_data = np.load(act_file, allow_pickle=True).item()
    act_dict = act_data["activation_function_ablation"]
    activations = list(act_dict.keys())
    
    # Accuracy curves
    fig_act_acc, ax_act_acc = plt.subplots(figsize=(8,6))
    for act in activations:
        acc_train = act_dict[act]["metrics"]["train"]
        acc_val = act_dict[act]["metrics"]["val"]
        ax_act_acc.plot(range(1, len(acc_train)+1), acc_train, marker="o", label=f"{act} - Train")
        ax_act_acc.plot(range(1, len(acc_val)+1), acc_val, marker="o", linestyle="--", label=f"{act} - Val")
    ax_act_acc.set_title("Activation Function: Training & Validation Accuracy")
    ax_act_acc.set_xlabel("Epochs")
    ax_act_acc.set_ylabel("Accuracy")
    ax_act_acc.legend()
    fig_act_acc.tight_layout()
    fig_act_acc.savefig(os.path.join("figures", "ablation_activation_accuracy.png"), dpi=300)
    plt.close(fig_act_acc)
    
    # Loss curves
    fig_act_loss, ax_act_loss = plt.subplots(figsize=(8,6))
    for act in activations:
        loss_train = act_dict[act]["losses"]["train"]
        loss_val = act_dict[act]["losses"]["val"]
        ax_act_loss.plot(range(1, len(loss_train)+1), loss_train, marker="o", label=f"{act} - Train")
        ax_act_loss.plot(range(1, len(loss_val)+1), loss_val, marker="o", linestyle="--", label=f"{act} - Val")
    ax_act_loss.set_title("Activation Function: Training & Validation Loss")
    ax_act_loss.set_xlabel("Epochs")
    ax_act_loss.set_ylabel("Loss")
    ax_act_loss.legend()
    fig_act_loss.tight_layout()
    fig_act_loss.savefig(os.path.join("figures", "ablation_activation_loss.png"), dpi=300)
    plt.close(fig_act_loss)
    print("Activation Function Ablation plots saved.")
except Exception as e:
    print("Error in Activation Function Ablation plots:", e)


# ---------------------------
# 5. ABLATION: Regularization Technique Ablation
#    Aggregate curves for regularization methods.
# ---------------------------
try:
    reg_file = "experiment_results/experiment_34a7df8801234f33b075ddc1909e9dda_proc_2520697/experiment_data.npy"
    reg_data = np.load(reg_file, allow_pickle=True).item()
    reg_dict = reg_data["regularization"]
    methods = list(reg_dict.keys())
    
    # Metrics curves
    fig_reg_acc, ax_reg_acc = plt.subplots(figsize=(8,6))
    for m in methods:
        met_train = reg_dict[m]["metrics"]["train"]
        met_val = reg_dict[m]["metrics"]["val"]
        ax_reg_acc.plot(range(1, len(met_train)+1), met_train, marker="o", label=f"{m} - Train")
        ax_reg_acc.plot(range(1, len(met_val)+1), met_val, marker="o", linestyle="--", label=f"{m} - Val")
    ax_reg_acc.set_title("Regularization: Training & Validation Metric")
    ax_reg_acc.set_xlabel("Epochs")
    ax_reg_acc.set_ylabel("Metric Value")
    ax_reg_acc.legend()
    fig_reg_acc.tight_layout()
    fig_reg_acc.savefig(os.path.join("figures", "ablation_regularization_metric.png"), dpi=300)
    plt.close(fig_reg_acc)
    
    # Loss curves
    fig_reg_loss, ax_reg_loss = plt.subplots(figsize=(8,6))
    for m in methods:
        loss_train = reg_dict[m]["losses"]["train"]
        loss_val = reg_dict[m]["losses"]["val"]
        ax_reg_loss.plot(range(1, len(loss_train)+1), loss_train, marker="o", label=f"{m} - Train")
        ax_reg_loss.plot(range(1, len(loss_val)+1), loss_val, marker="o", linestyle="--", label=f"{m} - Val")
    ax_reg_loss.set_title("Regularization: Training & Validation Loss")
    ax_reg_loss.set_xlabel("Epochs")
    ax_reg_loss.set_ylabel("Loss")
    ax_reg_loss.legend()
    fig_reg_loss.tight_layout()
    fig_reg_loss.savefig(os.path.join("figures", "ablation_regularization_loss.png"), dpi=300)
    plt.close(fig_reg_loss)
    print("Regularization Ablation plots saved.")
except Exception as e:
    print("Error in Regularization Ablation plots:", e)


# ---------------------------
# 6. ABLATION: Input Dimensionality Reduction Ablation
#    Compare original data and PCA-reduced data metrics and losses.
# ---------------------------
try:
    ind_file = "experiment_results/experiment_939d41a3dc1d4096a5732bee9bf1d874_proc_2520695/experiment_data.npy"
    ind_data = np.load(ind_file, allow_pickle=True).item()
    orig = ind_data["input_dimensionality_reduction"]["original_data"]
    red  = ind_data["input_dimensionality_reduction"]["reduced_data"]
    epochs = range(1, len(orig["metrics"]["train"]) + 1)
    
    fig_ind, axes = plt.subplots(2, 2, figsize=(12,10))
    # Original accuracy
    axes[0,0].plot(epochs, orig["metrics"]["train"], marker="o", label="Train")
    axes[0,0].plot(epochs, orig["metrics"]["val"], marker="o", linestyle="--", label="Val")
    axes[0,0].set_title("Original Data Accuracy")
    axes[0,0].set_xlabel("Epochs")
    axes[0,0].set_ylabel("Accuracy")
    axes[0,0].legend()
    # Original loss
    axes[0,1].plot(epochs, orig["losses"]["train"], marker="o", label="Train")
    axes[0,1].plot(epochs, orig["losses"]["val"], marker="o", linestyle="--", label="Val")
    axes[0,1].set_title("Original Data Loss")
    axes[0,1].set_xlabel("Epochs")
    axes[0,1].set_ylabel("Loss")
    axes[0,1].legend()
    # Reduced accuracy
    axes[1,0].plot(epochs, red["metrics"]["train"], marker="o", label="Train")
    axes[1,0].plot(epochs, red["metrics"]["val"], marker="o", linestyle="--", label="Val")
    axes[1,0].set_title("Reduced Data Accuracy")
    axes[1,0].set_xlabel("Epochs")
    axes[1,0].set_ylabel("Accuracy")
    axes[1,0].legend()
    # Reduced loss
    axes[1,1].plot(epochs, red["losses"]["train"], marker="o", label="Train")
    axes[1,1].plot(epochs, red["losses"]["val"], marker="o", linestyle="--", label="Val")
    axes[1,1].set_title("Reduced Data Loss")
    axes[1,1].set_xlabel("Epochs")
    axes[1,1].set_ylabel("Loss")
    axes[1,1].legend()
    
    plt.tight_layout()
    fig_ind.savefig(os.path.join("figures", "ablation_input_dimensionality.png"), dpi=300)
    plt.close(fig_ind)
    print("Input Dimensionality Reduction Ablation plot saved.")
except Exception as e:
    print("Error in Input Dimensionality Reduction Ablation plot:", e)


# ---------------------------
# 7. ABLATION: Batch Size Variation Ablation
#    Aggregate training and validation curves.
# ---------------------------
try:
    bs_file = "experiment_results/experiment_283449b903594a63b4555c42505b3dc0_proc_2520696/experiment_data.npy"
    bs_data = np.load(bs_file, allow_pickle=True).item()
    bs_section = bs_data["batch_size_ablation"]["RQS"]
    epochs = range(1, len(bs_section["losses"]["train"]) + 1)

    fig_bs, (ax_bs1, ax_bs2) = plt.subplots(1, 2, figsize=(12,5))
    ax_bs1.plot(epochs, bs_section["losses"]["train"], marker="o", label="Training Loss")
    ax_bs1.plot(epochs, bs_section["losses"]["val"], marker="o", label="Validation Loss")
    ax_bs1.set_title("Batch Size: Training vs Validation Loss")
    ax_bs1.set_xlabel("Epochs")
    ax_bs1.set_ylabel("Loss")
    ax_bs1.legend()
    
    ax_bs2.plot(epochs, bs_section["metrics"]["train"], marker="o", label="Training Metric")
    ax_bs2.plot(epochs, bs_section["metrics"]["val"], marker="o", label="Validation Metric")
    ax_bs2.set_title("Batch Size: Training vs Validation Metric")
    ax_bs2.set_xlabel("Epochs")
    ax_bs2.set_ylabel("Metric")
    ax_bs2.legend()
    
    plt.tight_layout()
    fig_bs.savefig(os.path.join("figures", "ablation_batch_size.png"), dpi=300)
    plt.close(fig_bs)
    print("Batch Size Ablation plot saved.")
except Exception as e:
    print("Error in Batch Size Ablation plot:", e)


# ---------------------------
# 8. ABLATION: Data Distribution Variation Ablation
#    Aggregate plots for normal, uniform and exponential distributions.
# ---------------------------
try:
    dd_file = "experiment_results/experiment_338470271afb4a4b86cab398f2e6d7c1_proc_2520697/experiment_data.npy"
    dd_data = np.load(dd_file, allow_pickle=True).item()
    dd_section = dd_data["data_dist_variation"]
    distributions = ["normal", "uniform", "exponential"]
    
    # Accuracy plot
    fig_dd_acc, ax_dd_acc = plt.subplots(figsize=(8,6))
    for dist in distributions:
        acc_train = dd_section[dist]["metrics"]["train"]
        acc_val = dd_section[dist]["metrics"]["val"]
        ax_dd_acc.plot(range(1, len(acc_train)+1), acc_train, marker="o", label=f"{dist.capitalize()} Train")
        ax_dd_acc.plot(range(1, len(acc_val)+1), acc_val, marker="o", linestyle="--", label=f"{dist.capitalize()} Val")
    ax_dd_acc.set_title("Data Distribution: Accuracy")
    ax_dd_acc.set_xlabel("Epochs")
    ax_dd_acc.set_ylabel("Accuracy")
    ax_dd_acc.legend()
    fig_dd_acc.tight_layout()
    fig_dd_acc.savefig(os.path.join("figures", "ablation_data_distribution_accuracy.png"), dpi=300)
    plt.close(fig_dd_acc)
    
    # Loss plot
    fig_dd_loss, ax_dd_loss = plt.subplots(figsize=(8,6))
    for dist in distributions:
        loss_train = dd_section[dist]["losses"]["train"]
        loss_val = dd_section[dist]["losses"]["val"]
        ax_dd_loss.plot(range(1, len(loss_train)+1), loss_train, marker="o", label=f"{dist.capitalize()} Train")
        ax_dd_loss.plot(range(1, len(loss_val)+1), loss_val, marker="o", linestyle="--", label=f"{dist.capitalize()} Val")
    ax_dd_loss.set_title("Data Distribution: Loss")
    ax_dd_loss.set_xlabel("Epochs")
    ax_dd_loss.set_ylabel("Loss")
    ax_dd_loss.legend()
    fig_dd_loss.tight_layout()
    fig_dd_loss.savefig(os.path.join("figures", "ablation_data_distribution_loss.png"), dpi=300)
    plt.close(fig_dd_loss)
    print("Data Distribution Variation Ablation plots saved.")
except Exception as e:
    print("Error in Data Distribution Variation Ablation plots:", e)


# ---------------------------
# 9. ABLATION: Multi-Dataset Evaluation Ablation
#    Aggregate accuracy and loss curves from multi-dataset evaluation.
# ---------------------------
try:
    mde_file = "experiment_results/experiment_4914897147ff4bf683b77282574e3e6e_proc_2520695/experiment_data.npy"
    mde_data = np.load(mde_file, allow_pickle=True).item()
    mde_section = mde_data["multi_dataset_evaluation"]
    datasets = list(mde_section.keys())
    
    # Accuracy curves
    fig_mde_acc, ax_mde_acc = plt.subplots(figsize=(8,6))
    for dset in datasets:
        acc_train = mde_section[dset]["metrics"]["train"]
        acc_val = mde_section[dset]["metrics"]["val"]
        ax_mde_acc.plot(range(1, len(acc_train)+1), acc_train, marker="o", label=f"{dset} Train")
        ax_mde_acc.plot(range(1, len(acc_val)+1), acc_val, marker="o", linestyle="--", label=f"{dset} Val")
    ax_mde_acc.set_title("Multi-Dataset Evaluation: Accuracy")
    ax_mde_acc.set_xlabel("Epochs")
    ax_mde_acc.set_ylabel("Accuracy")
    ax_mde_acc.legend()
    fig_mde_acc.tight_layout()
    fig_mde_acc.savefig(os.path.join("figures", "ablation_multi_dataset_accuracy.png"), dpi=300)
    plt.close(fig_mde_acc)
    
    # Loss curves
    fig_mde_loss, ax_mde_loss = plt.subplots(figsize=(8,6))
    for dset in datasets:
        loss_train = mde_section[dset]["losses"]["train"]
        loss_val = mde_section[dset]["losses"]["val"]
        ax_mde_loss.plot(range(1, len(loss_train)+1), loss_train, marker="o", label=f"{dset} Train")
        ax_mde_loss.plot(range(1, len(loss_val)+1), loss_val, marker="o", linestyle="--", label=f"{dset} Val")
    ax_mde_loss.set_title("Multi-Dataset Evaluation: Loss")
    ax_mde_loss.set_xlabel("Epochs")
    ax_mde_loss.set_ylabel("Loss")
    ax_mde_loss.legend()
    fig_mde_loss.tight_layout()
    fig_mde_loss.savefig(os.path.join("figures", "ablation_multi_dataset_loss.png"), dpi=300)
    plt.close(fig_mde_loss)
    print("Multi-Dataset Evaluation Ablation plots saved.")
except Exception as e:
    print("Error in Multi-Dataset Evaluation Ablation plots:", e)


# ---------------------------
# 10. ABLATION: Multiple Synthetic Dataset Evaluation
#    For each synthetic dataset, plot the training and validation curves side-by-side.
# ---------------------------
try:
    msd_file = "experiment_results/experiment_f8a8159614684d1098a83a7b7d66e638_proc_2520696/experiment_data.npy"
    msd_data = np.load(msd_file, allow_pickle=True).item()
    msd_section = msd_data["multiple_synthetic_datasets"]
    # We aggregate by creating one figure with 3 subplots (if there are three datasets)
    n_datasets = len(msd_section)
    fig_msd, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets,5))
    if n_datasets == 1:
        axes = [axes]  # Make it iterable if only one dataset
    for i, dset in enumerate(msd_section.keys()):
        data = msd_section[dset]
        epochs = range(1, len(data["metrics"]["train"]) + 1)
        # Plot both metric and loss
        axes[i].plot(epochs, data["metrics"]["train"], marker="o", label="Train Metric")
        axes[i].plot(epochs, data["metrics"]["val"], marker="o", linestyle="--", label="Val Metric")
        axes[i].plot(epochs, data["losses"]["train"], marker="s", label="Train Loss")
        axes[i].plot(epochs, data["losses"]["val"], marker="s", linestyle="--", label="Val Loss")
        axes[i].set_title(f"{dset} Curves")
        axes[i].set_xlabel("Epochs")
        axes[i].legend()
    plt.tight_layout()
    fig_msd.savefig(os.path.join("figures", "ablation_multiple_synthetic.png"), dpi=300)
    plt.close(fig_msd)
    print("Multiple Synthetic Dataset Evaluation plot saved.")
except Exception as e:
    print("Error in Multiple Synthetic Dataset Evaluation plot:", e)

print("All plots generated.")