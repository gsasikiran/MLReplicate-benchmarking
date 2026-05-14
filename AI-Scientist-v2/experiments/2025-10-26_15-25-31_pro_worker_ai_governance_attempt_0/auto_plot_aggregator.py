#!/usr/bin/env python3
"""
Final aggregator script for "Reimagining AI Safety: A Pro-Worker Framework for the Future of Work"
This script loads experiment data from .npy files and generates 12 unique, publication-quality figures,
saved in the "figures" directory.
Each plotting section is wrapped in a try/except block to ensure robust execution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Use a universally available style, for example "ggplot"
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 14})

# Create output directory
os.makedirs("figures", exist_ok=True)

# Helper function for aesthetics (no underscores in labels)
def adjust_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=14)

# Paths to experiment data files
baseline_path = "experiment_results/experiment_57a2c1c148c64a0f930e82c31d60a756_proc_2531281/experiment_data.npy"
feature_count_path = "experiment_results/experiment_2298b1976964473091b3fa45b1349515_proc_2532100/experiment_data.npy"
reg_path = "experiment_results/experiment_b24130739be5457aa1e8627608eed79c_proc_2532100/experiment_data.npy"
epochs_path = "experiment_results/experiment_2acca6b06fe941ae83366003f9b721ca_proc_2532101/experiment_data.npy"

# Load experiment data dictionaries, using try/except so that one missing file does not stop execution.
try:
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading baseline data: {e}")
    baseline_data = {}

try:
    feature_data = np.load(feature_count_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading feature count ablation data: {e}")
    feature_data = {}

try:
    reg_data = np.load(reg_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading regularization data: {e}")
    reg_data = {}

try:
    epochs_data = np.load(epochs_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading training epochs ablation data: {e}")
    epochs_data = {}

###############################################################################
# Plot 1: Baseline Loss Curves for Different Learning Rates
try:
    base = baseline_data["hyperparam_tuning_learning_rate"]["synthetic_data"]
    learning_rates = base["learning_rates"]
    train_losses = base["losses"]["train"]
    val_losses = base["losses"]["val"]

    plt.figure(dpi=300, figsize=(8,6))
    for i, lr in enumerate(learning_rates):
        plt.plot(train_losses[i], label=f"Train Loss (LR = {lr})", linewidth=2)
        plt.plot(val_losses[i], label=f"Validation Loss (LR = {lr})", linestyle="--", linewidth=2)
    plt.title("Baseline Loss Curves Across Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    adjust_ax(plt.gca())
    plt.savefig(os.path.join("figures", "Baseline Loss Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 1 (Baseline Loss Curves): {e}")
    plt.close()

###############################################################################
# Plot 2: Baseline Accuracy Curves for Different Learning Rates
try:
    train_accuracy = base["metrics"]["train"]
    val_accuracy = base["metrics"]["val"]

    plt.figure(dpi=300, figsize=(8,6))
    for i, lr in enumerate(learning_rates):
        plt.plot(train_accuracy[i], label=f"Train Accuracy (LR = {lr})", linewidth=2)
        plt.plot(val_accuracy[i], label=f"Validation Accuracy (LR = {lr})", linestyle="--", linewidth=2)
    plt.title("Baseline Accuracy Curves Across Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    adjust_ax(plt.gca())
    plt.savefig(os.path.join("figures", "Baseline Accuracy Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 2 (Baseline Accuracy Curves): {e}")
    plt.close()

###############################################################################
# Plot 3: Feature Count Ablation Loss Curves (one subplot per feature count)
try:
    feat = feature_data["feature_count_ablation"]["synthetic_data"]
    feature_counts = feat["feature_counts"]
    losses_train_feat = feat["losses"]["train"]
    losses_val_feat = feat["losses"]["val"]
    n_feat = len(feature_counts)

    fig, axes = plt.subplots(1, n_feat, figsize=(5*n_feat, 5), dpi=300)
    if n_feat == 1:
        axes = [axes]  # Ensure iterable if only one plot
    for idx, fc in enumerate(feature_counts):
        axes[idx].plot(losses_train_feat[idx], label="Train Loss", linewidth=2)
        axes[idx].plot(losses_val_feat[idx], label="Validation Loss", linestyle="--", linewidth=2)
        axes[idx].set_title(f"Loss Curves with {fc} Features")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss")
        axes[idx].legend()
        adjust_ax(axes[idx])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Feature Count Ablation Loss Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 3 (Feature Count Ablation Loss Curves): {e}")
    plt.close()

###############################################################################
# Plot 4: Feature Count Ablation Accuracy Curves (one subplot per feature count)
try:
    accuracy_train_feat = feat["metrics"]["train"]
    accuracy_val_feat = feat["metrics"]["val"]
    n_feat = len(feature_counts)

    fig, axes = plt.subplots(1, n_feat, figsize=(5*n_feat, 5), dpi=300)
    if n_feat == 1:
        axes = [axes]
    for idx, fc in enumerate(feature_counts):
        axes[idx].plot(accuracy_train_feat[idx], label="Train Accuracy", linewidth=2)
        axes[idx].plot(accuracy_val_feat[idx], label="Validation Accuracy", linestyle="--", linewidth=2)
        axes[idx].set_title(f"Accuracy Curves with {fc} Features")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Accuracy")
        axes[idx].legend()
        adjust_ax(axes[idx])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Feature Count Ablation Accuracy Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 4 (Feature Count Ablation Accuracy Curves): {e}")
    plt.close()

###############################################################################
# Plot 5: Regularization Loss Curves (L2 Regularization vs Dropout) in one figure
try:
    l2 = reg_data["L2_regularization"]["synthetic_data"]
    dropout = reg_data["dropout"]["synthetic_data"]

    fig, axes = plt.subplots(1, 2, figsize=(12,5), dpi=300)
    # L2 Regularization
    axes[0].plot(l2["losses"]["train"], label="Train Loss", linewidth=2)
    axes[0].plot(l2["losses"]["val"], label="Validation Loss", linestyle="--", linewidth=2)
    axes[0].set_title("L2 Regularization Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    adjust_ax(axes[0])
    # Dropout Regularization
    axes[1].plot(dropout["losses"]["train"], label="Train Loss", linewidth=2)
    axes[1].plot(dropout["losses"]["val"], label="Validation Loss", linestyle="--", linewidth=2)
    axes[1].set_title("Dropout Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    adjust_ax(axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Regularization Loss Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 5 (Regularization Loss Curves): {e}")
    plt.close()

###############################################################################
# Plot 6: Regularization Accuracy Curves (L2 Regularization vs Dropout) in one figure
try:
    fig, axes = plt.subplots(1, 2, figsize=(12,5), dpi=300)
    # L2 Regularization
    axes[0].plot(l2["metrics"]["train"], label="Train Accuracy", linewidth=2)
    axes[0].plot(l2["metrics"]["val"], label="Validation Accuracy", linestyle="--", linewidth=2)
    axes[0].set_title("L2 Regularization Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    adjust_ax(axes[0])
    # Dropout Regularization
    axes[1].plot(dropout["metrics"]["train"], label="Train Accuracy", linewidth=2)
    axes[1].plot(dropout["metrics"]["val"], label="Validation Accuracy", linestyle="--", linewidth=2)
    axes[1].set_title("Dropout Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    adjust_ax(axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Regularization Accuracy Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 6 (Regularization Accuracy Curves): {e}")
    plt.close()

###############################################################################
# Plot 7: Training Epochs Ablation Loss Curves
try:
    epochs = epochs_data["training_epochs_ablation"]["synthetic_data"]
    te_train_loss = epochs["losses"]["train"]
    te_val_loss = epochs["losses"]["val"]

    plt.figure(dpi=300, figsize=(8,6))
    plt.plot(te_train_loss, label="Train Loss", linewidth=2)
    plt.plot(te_val_loss, label="Validation Loss", linestyle="--", linewidth=2)
    plt.title("Training Epochs Ablation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    adjust_ax(plt.gca())
    plt.savefig(os.path.join("figures", "Training Epochs Ablation Loss Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 7 (Training Epochs Ablation Loss Curves): {e}")
    plt.close()

###############################################################################
# Plot 8: Training Epochs Ablation Accuracy Curves
try:
    te_train_acc = epochs["metrics"]["train"]
    te_val_acc = epochs["metrics"]["val"]

    plt.figure(dpi=300, figsize=(8,6))
    plt.plot(te_train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(te_val_acc, label="Validation Accuracy", linestyle="--", linewidth=2)
    plt.title("Training Epochs Ablation Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    adjust_ax(plt.gca())
    plt.savefig(os.path.join("figures", "Training Epochs Ablation Accuracy Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 8 (Training Epochs Ablation Accuracy Curves): {e}")
    plt.close()

###############################################################################
# Plot 9: Final Validation Accuracy Comparison Across Experiments
try:
    # Baseline: final validation accuracy per learning rate
    final_base_acc = [vals[-1] for vals in val_accuracy]
    base_labels = [f"LR {lr}" for lr in learning_rates]
    # Feature Count: final validation accuracy per feature count
    final_feat_acc = [vals[-1] for vals in accuracy_val_feat]
    feat_labels = [f"{fc} Feat" for fc in feature_counts]
    # Regularization: final validation accuracy for L2 and Dropout
    final_reg_acc = [l2["metrics"]["val"][-1], dropout["metrics"]["val"][-1]]
    reg_labels = ["L2", "Dropout"]
    # Epochs Ablation: final validation accuracy
    final_epochs_acc = te_val_acc[-1]
    epochs_label = "Epoch Ablation"
    # Aggregate labels and values
    all_labels = base_labels + feat_labels + reg_labels + [epochs_label]
    all_values = final_base_acc + final_feat_acc + final_reg_acc + [final_epochs_acc]

    plt.figure(dpi=300, figsize=(10,6))
    plt.plot(range(len(all_values)), all_values, marker="o", linestyle="-", linewidth=2)
    plt.xticks(range(len(all_values)), all_labels, rotation=45)
    plt.title("Final Validation Accuracy Comparison")
    plt.xlabel("Experiment Setting")
    plt.ylabel("Validation Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    adjust_ax(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Final Validation Accuracy Comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 9 (Final Validation Accuracy Comparison): {e}")
    plt.close()

###############################################################################
# Plot 10: Final Validation Loss Comparison Across Experiments
try:
    # Baseline: final validation loss per learning rate
    final_base_loss = [vals[-1] for vals in val_losses]
    base_loss_labels = [f"LR {lr}" for lr in learning_rates]
    # Feature Count: final validation loss per feature count
    final_feat_loss = [vals[-1] for vals in losses_val_feat]
    feat_loss_labels = [f"{fc} Feat" for fc in feature_counts]
    # Regularization: final validation loss for L2 and Dropout
    final_reg_loss = [l2["losses"]["val"][-1], dropout["losses"]["val"][-1]]
    reg_loss_labels = ["L2", "Dropout"]
    # Epochs Ablation: final validation loss
    final_epochs_loss = te_val_loss[-1]
    epochs_loss_label = "Epoch Ablation"
    # Aggregate
    all_loss_labels = base_loss_labels + feat_loss_labels + reg_loss_labels + [epochs_loss_label]
    all_loss_values = final_base_loss + final_feat_loss + final_reg_loss + [final_epochs_loss]

    plt.figure(dpi=300, figsize=(10,6))
    plt.plot(range(len(all_loss_values)), all_loss_values, marker="s", linestyle="-", linewidth=2, color="darkred")
    plt.xticks(range(len(all_loss_values)), all_loss_labels, rotation=45)
    plt.title("Final Validation Loss Comparison")
    plt.xlabel("Experiment Setting")
    plt.ylabel("Validation Loss")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    adjust_ax(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Final Validation Loss Comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 10 (Final Validation Loss Comparison): {e}")
    plt.close()

###############################################################################
# Plot 11: Improvement Ratio Comparison
try:
    # Improvement ratio = (initial loss - final loss) / initial loss
    base_ratios = [ (vals[0]-vals[-1])/vals[0] if vals[0]!=0 else 0 for vals in val_losses ]
    feat_ratios = [ (vals[0]-vals[-1])/vals[0] if vals[0]!=0 else 0 for vals in losses_val_feat ]
    l2_ratio = (l2["losses"]["val"][0]-l2["losses"]["val"][-1])/l2["losses"]["val"][0] if l2["losses"]["val"][0]!=0 else 0
    dropout_ratio = (dropout["losses"]["val"][0]-dropout["losses"]["val"][-1])/dropout["losses"]["val"][0] if dropout["losses"]["val"][0]!=0 else 0
    epoch_ratio = (te_val_loss[0]-te_val_loss[-1])/te_val_loss[0] if te_val_loss[0]!=0 else 0

    all_ratio_labels = base_labels + feat_labels + ["L2", "Dropout", "Epoch Ablation"]
    all_ratios = base_ratios + feat_ratios + [l2_ratio, dropout_ratio, epoch_ratio]

    plt.figure(dpi=300, figsize=(10,6))
    plt.plot(range(len(all_ratios)), all_ratios, marker="D", linestyle="-", linewidth=2, color="green")
    plt.xticks(range(len(all_ratios)), all_ratio_labels, rotation=45)
    plt.title("Improvement Ratio Comparison")
    plt.xlabel("Experiment Setting")
    plt.ylabel("Improvement Ratio")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    adjust_ax(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Improvement Ratio Comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 11 (Improvement Ratio Comparison): {e}")
    plt.close()

###############################################################################
# Plot 12: Loss Improvement Dynamics Over Epochs (Aggregated)
try:
    # For baseline, compute normalized improvement curves for each learning rate and average them
    improvements = []
    for series in val_losses:
        improvements.append([ (series[0]-x)/series[0] if series[0]!=0 else 0 for x in series])
    avg_improvement = np.mean(improvements, axis=0)
    # For regularization, use L2 improvement curve
    l2_improvement = [ (l2["losses"]["val"][0]-x)/l2["losses"]["val"][0] if l2["losses"]["val"][0]!=0 else 0 for x in l2["losses"]["val"] ]
    # For epochs, compute improvement curve
    epochs_improvement = [ (te_val_loss[0]-x)/te_val_loss[0] if te_val_loss[0]!=0 else 0 for x in te_val_loss ]

    epochs_range = range(len(avg_improvement))
    fig, axes = plt.subplots(1, 3, figsize=(18,5), dpi=300)
    axes[0].plot(epochs_range, avg_improvement, marker="o", linewidth=2, color="blue")
    axes[0].set_title("Baseline Loss Improvement")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Normalized Improvement")
    adjust_ax(axes[0])
    axes[1].plot(range(len(l2_improvement)), l2_improvement, marker="s", linewidth=2, color="purple")
    axes[1].set_title("L2 Regularization Loss Improvement")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Normalized Improvement")
    adjust_ax(axes[1])
    axes[2].plot(range(len(epochs_improvement)), epochs_improvement, marker="^", linewidth=2, color="orange")
    axes[2].set_title("Training Epochs Ablation Improvement")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Normalized Improvement")
    adjust_ax(axes[2])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Loss Improvement Dynamics.png"))
    plt.close()
except Exception as e:
    print(f"Error in Plot 12 (Loss Improvement Dynamics): {e}")
    plt.close()

print("Final figures have been generated and saved in the 'figures' directory.")