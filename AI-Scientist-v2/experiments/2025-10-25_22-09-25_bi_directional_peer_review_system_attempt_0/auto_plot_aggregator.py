#!/usr/bin/env python3
"""
Final Aggregated Plotting Script for the Bi-Directional Peer Review System Paper

This script loads pre-saved .npy experiment results from the JSON summaries and produces
a complete set of final figures saved in the "figures/" directory. It aggregates results
from the baseline, ablation, and learning rate/model depth studies. Each figure is created
in its own try-except block so that errors in one do not prevent other figures from generating.

All plots are produced using Matplotlib with enhanced font sizes and professional style.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Set global plot style for publication quality
plt.rcParams.update({
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False
})

def style_axis(ax):
    """Helper to remove top/right spines and adjust tick parameters."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major")

# ---------------------------
# Plot 1: Baseline - Training Loss
# ---------------------------
try:
    # Load baseline experiment data (shared with research, same file path)
    baseline_file = "experiment_results/experiment_d53a2341777547a18be6dee17fbc923d_proc_2516354/experiment_data.npy"
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    # Retrieve training loss data from hyperparameter tuning section
    loss_arr = baseline_data["hyperparam_tuning_num_hidden_units"]["synthetic_data"]["losses"]["train"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(loss_arr)), loss_arr, label="Training Loss", color="blue", marker="o")
    ax.set_title("Baseline: Training Loss over Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    style_axis(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_training_loss.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 1 (Baseline Training Loss):", e)
    plt.close()

# ---------------------------
# Plot 2: Baseline - Training RQS Metric
# ---------------------------
try:
    rqs_arr = baseline_data["hyperparam_tuning_num_hidden_units"]["synthetic_data"]["metrics"]["train"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(rqs_arr)), rqs_arr, label="Reviewer Quality Score", color="green", marker="s")
    ax.set_title("Baseline: Training RQS over Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("RQS")
    style_axis(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "baseline_training_rqs.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 2 (Baseline Training RQS):", e)
    plt.close()

# ---------------------------
# Plot 3: Dataset Variability Impact - Training Loss per Synthetic Dataset
# (Aggregated in one figure with 3 subplots)
# ---------------------------
try:
    ablation_file1 = "experiment_results/experiment_8ce040b5e8504104a461f42f2475645d_proc_2517443/experiment_data.npy"
    ablation_data1 = np.load(ablation_file1, allow_pickle=True).item()
    dataset_keys = ["dataset_1", "dataset_2", "dataset_3"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, key in enumerate(dataset_keys):
        losses = ablation_data1["dataset_variability_impact"][key]["losses"]["train"]
        axes[i].plot(range(len(losses)), losses, label="Training Loss", color="purple")
        axes[i].set_title(f"{key.replace('_', ' ').title()} Loss")
        axes[i].set_xlabel("Epochs")
        if i==0:
            axes[i].set_ylabel("Loss")
        style_axis(axes[i])
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "dataset_variability_training_loss.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 3 (Dataset Variability Training Loss):", e)
    plt.close()

# ---------------------------
# Plot 4: Dataset Variability Impact - RQS Metric per Synthetic Dataset
# (Aggregated in one figure with 3 subplots)
# ---------------------------
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, key in enumerate(dataset_keys):
        rqs = ablation_data1["dataset_variability_impact"][key]["metrics"]["train"]
        axes[i].plot(range(len(rqs)), rqs, label="RQS", color="darkorange")
        axes[i].set_title(f"{key.replace('_', ' ').title()} RQS")
        axes[i].set_xlabel("Epochs")
        if i==0:
            axes[i].set_ylabel("RQS")
        style_axis(axes[i])
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "dataset_variability_training_rqs.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 4 (Dataset Variability RQS):", e)
    plt.close()

# ---------------------------
# Plot 5: Impact of Batch Size - Overlaid Training Loss Curves
# (Assuming the npy file contains dicts keyed by batch size as strings)
# ---------------------------
try:
    batch_file = "experiment_results/experiment_e1614dff1f264eef8742af79474fa2d0_proc_2517445/experiment_data.npy"
    batch_data = np.load(batch_file, allow_pickle=True).item()
    # Assume structure is a dict with keys "16", "32", "64", "128" under losses/train
    loss_dict = batch_data["impact_of_batch_size"]["synthetic_data"]["losses"]["train"]
    batch_sizes = [16, 32, 64, 128]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for bs in batch_sizes:
        # Use string key if available, else try index with bs
        key = str(bs)
        # If not dict, then use the same array for all (fallback)
        curve = loss_dict[key] if isinstance(loss_dict, dict) and key in loss_dict else loss_dict
        ax.plot(range(len(curve)), curve, marker="o", label=f"Batch Size {bs}")
    ax.set_title("Impact of Batch Size: Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    style_axis(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "batch_size_training_loss_overlay.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 5 (Batch Size Training Loss Overlay):", e)
    plt.close()

# ---------------------------
# Plot 6: Impact of Batch Size - Overlaid RQS Curves
# ---------------------------
try:
    rqs_dict = batch_data["impact_of_batch_size"]["synthetic_data"]["metrics"]["train"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for bs in batch_sizes:
        key = str(bs)
        curve = rqs_dict[key] if isinstance(rqs_dict, dict) and key in rqs_dict else rqs_dict
        ax.plot(range(len(curve)), curve, marker="s", label=f"Batch Size {bs}")
    ax.set_title("Impact of Batch Size: Training RQS")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("RQS")
    style_axis(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "batch_size_training_rqs_overlay.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 6 (Batch Size Training RQS Overlay):", e)
    plt.close()

# ---------------------------
# Plot 7: Impact of Learning Rate Variations - Composite Figure
# (3 subplots: Training Loss, RQS, and Predictions vs Ground Truth)
# ---------------------------
try:
    lr_file = "experiment_results/experiment_f761e7f24f54494383dd38d5c202247a_proc_2517444/experiment_data.npy"
    lr_data = np.load(lr_file, allow_pickle=True).item()
    lr_section = lr_data["learning_rate_variations"]["synthetic_data"]
    
    # Retrieve arrays
    lr_loss = lr_section["losses"]["train"]
    lr_rqs = lr_section["metrics"]["train"]
    predictions = np.concatenate(lr_section["predictions"])
    ground_truth = np.concatenate(lr_section["ground_truth"])
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    # Subplot 1: Loss
    axes[0].plot(range(len(lr_loss)), lr_loss, color="navy", marker="o", label="Loss")
    axes[0].set_title("Learning Rate Variations: Training Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    style_axis(axes[0])
    axes[0].legend()
    
    # Subplot 2: RQS
    axes[1].plot(range(len(lr_rqs)), lr_rqs, color="teal", marker="s", label="RQS")
    axes[1].set_title("Learning Rate Variations: Training RQS")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("RQS")
    style_axis(axes[1])
    axes[1].legend()

    # Subplot 3: Predictions vs Ground Truth Scatter
    axes[2].scatter(ground_truth, predictions, alpha=0.5, color="maroon", label="Predictions")
    axes[2].plot([ground_truth.min(), ground_truth.max()],
                 [ground_truth.min(), ground_truth.max()],
                 color="gray", linestyle="--", label="Ideal")
    axes[2].set_title("Predictions vs Ground Truth")
    axes[2].set_xlabel("Ground Truth")
    axes[2].set_ylabel("Predictions")
    style_axis(axes[2])
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "learning_rate_variations_composite.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 7 (Learning Rate Variations Composite):", e)
    plt.close()

# ---------------------------
# Plot 8: Impact of Model Depth on Performance - Training Loss by Depth
# (Overlaid curves; assuming the npy file contains a dict keyed by depth)
# ---------------------------
try:
    depth_file = "experiment_results/experiment_eb14299027dd49dfb00b60fac71d853c_proc_2517445/experiment_data.npy"
    depth_data = np.load(depth_file, allow_pickle=True).item()
    # Try to retrieve loss curves per depth; if structured as dict, iterate keys
    depth_loss = depth_data["impact_of_model_depth"]["synthetic_data"]["losses"]["train"]
    fig, ax = plt.subplots(figsize=(8, 6))
    if isinstance(depth_loss, dict):
        # Sort keys by depth (assumed to be "1", "2", "3")
        for depth_key in sorted(depth_loss, key=lambda x: int(x)):
            ax.plot(range(len(depth_loss[depth_key])), depth_loss[depth_key],
                    marker="o", label=f"Depth {depth_key}")
    else:
        # If not dict, plot single curve
        ax.plot(range(len(depth_loss)), depth_loss, marker="o", label="Depth")
    ax.set_title("Model Depth: Training Loss Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    style_axis(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "model_depth_training_loss.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 8 (Model Depth Training Loss):", e)
    plt.close()

# ---------------------------
# Plot 9: Impact of Model Depth on Performance - RQS by Depth
# (Overlaid curves; similar logic as Plot 8)
# ---------------------------
try:
    depth_rqs = depth_data["impact_of_model_depth"]["synthetic_data"]["metrics"]["train"]
    fig, ax = plt.subplots(figsize=(8, 6))
    if isinstance(depth_rqs, dict):
        for depth_key in sorted(depth_rqs, key=lambda x: int(x)):
            ax.plot(range(len(depth_rqs[depth_key])), depth_rqs[depth_key],
                    marker="s", label=f"Depth {depth_key}")
    else:
        ax.plot(range(len(depth_rqs)), depth_rqs, marker="s", label="Depth")
    ax.set_title("Model Depth: Training RQS Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("RQS")
    style_axis(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "model_depth_training_rqs.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 9 (Model Depth Training RQS):", e)
    plt.close()

# ---------------------------
# Plot 10: Multi-Synthetic Dataset Performance - Training Loss per Dataset
# (Subplots for each of linear, high-noise, and polynomial data)
# ---------------------------
try:
    multi_file = "experiment_results/experiment_572425a7c87e4a6c99833f015fb4502c_proc_2517444/experiment_data.npy"
    multi_data = np.load(multi_file, allow_pickle=True).item()
    multi_keys = list(multi_data["multi_synthetic_dataset_performance"].keys())
    
    fig, axes = plt.subplots(1, len(multi_keys), figsize=(6*len(multi_keys), 5), sharey=True)
    for i, key in enumerate(multi_keys):
        losses = multi_data["multi_synthetic_dataset_performance"][key]["losses"]["train"]
        axes[i].plot(range(len(losses)), losses, label="Training Loss", color="darkblue")
        axes[i].set_title(f"{key.replace('_', ' ').title()} Loss")
        axes[i].set_xlabel("Epochs")
        if i==0:
            axes[i].set_ylabel("Loss")
        style_axis(axes[i])
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multi_dataset_training_loss_subplots.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 10 (Multi-Synthetic Dataset Loss Subplots):", e)
    plt.close()

# ---------------------------
# Plot 11: Multi-Synthetic Dataset Performance - Overlaid Loss Curves
# (All datasets on one axis for direct comparison)
# ---------------------------
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in multi_keys:
        losses = multi_data["multi_synthetic_dataset_performance"][key]["losses"]["train"]
        ax.plot(range(len(losses)), losses, marker="o", label=key.replace('_', ' ').title())
    ax.set_title("Multi-Synthetic Dataset: Overlaid Training Loss Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    style_axis(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multi_dataset_loss_overlay.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 11 (Multi-Synthetic Dataset Overlay):", e)
    plt.close()

# ---------------------------
# Plot 12: Appendix - Model Depth Combined Comparison
# (Two-panel figure: top for Loss, bottom for RQS, overlaid by depth)
# ---------------------------
try:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    if isinstance(depth_loss, dict):
        for depth_key in sorted(depth_loss, key=lambda x: int(x)):
            ax1.plot(range(len(depth_loss[depth_key])), depth_loss[depth_key],
                     marker="o", label=f"Depth {depth_key}")
    else:
        ax1.plot(range(len(depth_loss)), depth_loss, marker="o", label="Depth")
    ax1.set_title("Appendix: Model Depth Training Loss Comparison")
    ax1.set_ylabel("Loss")
    style_axis(ax1)
    ax1.legend()

    if isinstance(depth_rqs, dict):
        for depth_key in sorted(depth_rqs, key=lambda x: int(x)):
            ax2.plot(range(len(depth_rqs[depth_key])), depth_rqs[depth_key],
                     marker="s", label=f"Depth {depth_key}")
    else:
        ax2.plot(range(len(depth_rqs)), depth_rqs, marker="s", label="Depth")
    ax2.set_title("Appendix: Model Depth Training RQS Comparison")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("RQS")
    style_axis(ax2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("figures", "appendix_model_depth_combined.png"), dpi=300)
    plt.close()
except Exception as e:
    print("Error in Plot 12 (Appendix Model Depth Combined):", e)
    plt.close()

print("All plots generated and saved in the 'figures/' directory.")