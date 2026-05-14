#!/usr/bin/env python
"""
Final Aggregated Plotting Script for the bi_directional_peer_review_system Paper
This script loads experiment data from the provided .npy files and produces final,
publication‐quality figures stored in the "figures/" directory.
Each plotting section is wrapped in a try/except block so that a failure in one
plot does not affect the others.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set up a professional plot style and larger font size.
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.dpi"] = 300

# Remove top and right spines helper:
def format_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax

# Ensure the output directory exists.
os.makedirs("figures", exist_ok=True)

##########################
# 1. Baseline Loss Curves
##########################
try:
    # Load baseline experiment data
    baseline_path = "experiment_results/experiment_d7d82627649043d2a55c29c0f6aafd7e_proc_2518155/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    losses = baseline_data.get("peer_review", {}).get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    
    if train_loss and val_loss:
        plt.figure()
        plt.plot(train_loss, label="Training Loss", marker="o")
        plt.plot(val_loss, label="Validation Loss", marker="o")
        plt.title("Baseline: Peer Review Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        ax = plt.gca()
        format_axes(ax)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "baseline_peer_review_loss_curves.png"))
        plt.close()
    else:
        print("Baseline loss data missing.")
except Exception as e:
    print(f"Error in Baseline Loss Curves: {e}")
    plt.close()

##################################################
# 2. Multi-Dataset Evaluation (Ablation) - Losses
##################################################
try:
    md_path = "experiment_results/experiment_85f8e3f22bf641d7b9b4051bcaac01df_proc_2518845/experiment_data.npy"
    md_data = np.load(md_path, allow_pickle=True).item()
    multi_eval = md_data.get("multi_dataset_eval", {})
    datasets = list(multi_eval.keys())
    
    if datasets:
        # Plot Loss Curves for each dataset in one figure (1 row, N columns)
        fig, axs = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
        if len(datasets) == 1:
            axs = [axs]
        for i, ds in enumerate(datasets):
            ds_data = multi_eval.get(ds, {})
            epochs = range(1, len(ds_data.get("losses", {}).get("train", [])) + 1)
            axs[i].plot(epochs, ds_data.get("losses", {}).get("train", []), label="Train Loss", marker="o")
            axs[i].plot(epochs, ds_data.get("losses", {}).get("val", []), label="Validation Loss", marker="o")
            axs[i].set_title(f"{ds} Loss Curves")
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel("Loss")
            axs[i].legend()
            format_axes(axs[i])
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "multi_dataset_loss_curves.png"))
        plt.close()

        # Also, Plot RQI (metrics) curves if available
        fig, axs = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
        if len(datasets) == 1:
            axs = [axs]
        for i, ds in enumerate(datasets):
            ds_data = multi_eval.get(ds, {})
            epochs = range(1, len(ds_data.get("metrics", {}).get("train", [])) + 1)
            axs[i].plot(epochs, ds_data.get("metrics", {}).get("train", []), label="Train RQI", marker="o")
            axs[i].plot(epochs, ds_data.get("metrics", {}).get("val", []), label="Validation RQI", marker="o")
            axs[i].set_title(f"{ds} RQI Curves")
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel("RQI")
            axs[i].legend()
            format_axes(axs[i])
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "multi_dataset_rqi_curves.png"))
        plt.close()
    else:
        print("No multi-dataset evaluation data available.")
except Exception as e:
    print(f"Error in Multi-Dataset Evaluation plots: {e}")
    plt.close()

#####################################################
# 3. Learning Rate Schedule Evaluation (Fixed vs. Schedule)
#####################################################
try:
    lr_path = "experiment_results/experiment_1505666cf0ad4fe1ae6300ab6acc523c_proc_2518848/experiment_data.npy"
    lr_data = np.load(lr_path, allow_pickle=True).item()
    lr_sched = lr_data.get("learning_rate_schedule", {})
    
    # Figure for Loss Curves: fixed and schedule in side-by-side subplots.
    fixed = lr_sched.get("fixed", {})
    schedule = lr_sched.get("schedule", {})
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Fixed LR Loss curves
    axs[0].plot(fixed.get("losses", {}).get("train", []), label="Train Loss", marker="o")
    axs[0].plot(fixed.get("losses", {}).get("val", []), label="Validation Loss", marker="o")
    axs[0].set_title("Fixed LR Loss Curves")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    format_axes(axs[0])
    
    # Scheduled LR Loss curves
    axs[1].plot(schedule.get("losses", {}).get("train", []), label="Train Loss", marker="o")
    axs[1].plot(schedule.get("losses", {}).get("val", []), label="Validation Loss", marker="o")
    axs[1].set_title("Scheduled LR Loss Curves")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    format_axes(axs[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "lr_schedule_loss_curves.png"))
    plt.close()

    # Figure for Predictions vs Ground Truth (scatter plots)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Fixed LR Predictions
    fixed_pred = fixed.get("predictions", [])
    fixed_gt = fixed.get("ground_truth", [])
    axs[0].scatter(fixed_gt, fixed_pred, alpha=0.5)
    axs[0].plot([0, 1], [0, 1], "r--", label="Ideal")
    axs[0].set_title("Fixed LR Predictions vs Ground Truth")
    axs[0].set_xlabel("Ground Truth")
    axs[0].set_ylabel("Predictions")
    axs[0].legend()
    format_axes(axs[0])
    
    # Scheduled LR Predictions
    sched_pred = schedule.get("predictions", [])
    sched_gt = schedule.get("ground_truth", [])
    axs[1].scatter(sched_gt, sched_pred, alpha=0.5)
    axs[1].plot([0, 1], [0, 1], "r--", label="Ideal")
    axs[1].set_title("Scheduled LR Predictions vs Ground Truth")
    axs[1].set_xlabel("Ground Truth")
    axs[1].set_ylabel("Predictions")
    axs[1].legend()
    format_axes(axs[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "lr_schedule_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error in Learning Rate Schedule Evaluation plots: {e}")
    plt.close()

#####################################################################
# 4. Synthetic Dataset Diversity Evaluation (Appendix Plots)
#####################################################################
try:
    synth_div_path = "experiment_results/experiment_1d3e47a1f32645f89daf6d35c8fe2123_proc_2518846/experiment_data.npy"
    synth_div_data = np.load(synth_div_path, allow_pickle=True).item()
    # Distributions: uniform, normal, skewed
    distributions = ["uniform", "normal", "skewed"]
    
    # Loss Curves: create one row with three subplots.
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, dist in enumerate(distributions):
        dist_data = synth_div_data.get("ablation_study", {}).get(dist, {})
        epochs = range(1, len(dist_data.get("losses", {}).get("train", [])) + 1)
        axs[i].plot(epochs, dist_data.get("losses", {}).get("train", []), label="Train Loss", marker="o")
        axs[i].plot(epochs, dist_data.get("losses", {}).get("val", []), label="Validation Loss", marker="o")
        axs[i].set_title(f"{dist.capitalize()} Loss Curves")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        format_axes(axs[i])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "synthetic_diversity_losses.png"))
    plt.close()
    
    # Predictions: scatter plots for each distribution.
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, dist in enumerate(distributions):
        dist_data = synth_div_data.get("ablation_study", {}).get(dist, {})
        axs[i].scatter(dist_data.get("ground_truth", []), dist_data.get("predictions", []), alpha=0.5)
        axs[i].plot([min(dist_data.get("ground_truth", [0])), max(dist_data.get("ground_truth", [1]))],
                    [min(dist_data.get("ground_truth", [0])), max(dist_data.get("ground_truth", [1]))],
                    "r--", label="Ideal")
        axs[i].set_title(f"{dist.capitalize()} Predictions")
        axs[i].set_xlabel("Ground Truth")
        axs[i].set_ylabel("Predictions")
        axs[i].legend()
        format_axes(axs[i])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "synthetic_diversity_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error in Synthetic Dataset Diversity Evaluation plots: {e}")
    plt.close()

###############################################################
# 5. Activation Function Comparison (Ablation) - Losses and RQI
###############################################################
try:
    act_path = "experiment_results/experiment_664e44326ef24376aa76b6f306d9e06c_proc_2518845/experiment_data.npy"
    act_data = np.load(act_path, allow_pickle=True).item()
    act_comp = act_data.get("activation_function_comparison", {})
    losses_dict = act_comp.get("losses", {})
    metrics_dict = act_comp.get("metrics", {})
    activations = list(losses_dict.keys())
    
    # Loss Curves for each activation function in one figure.
    fig, axs = plt.subplots(1, len(activations), figsize=(5*len(activations), 4))
    if len(activations) == 1:
        axs = [axs]
    for i, act_name in enumerate(activations):
        ds = losses_dict.get(act_name, {})
        epochs = range(1, len(ds.get("train", [])) + 1)
        axs[i].plot(epochs, ds.get("train", []), label="Train Loss", marker="o")
        axs[i].plot(epochs, ds.get("val", []), label="Validation Loss", marker="o")
        axs[i].set_title(f"{act_name} Losses")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        format_axes(axs[i])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "activation_losses.png"))
    plt.close()
    
    # RQI Metrics for each activation function.
    fig, axs = plt.subplots(1, len(activations), figsize=(5*len(activations), 4))
    if len(activations) == 1:
        axs = [axs]
    for i, act_name in enumerate(activations):
        ds = metrics_dict.get(act_name, {})
        epochs = range(1, len(ds.get("train", [])) + 1)
        axs[i].plot(epochs, ds.get("train", []), label="RQI (Train)", marker="o")
        axs[i].set_title(f"{act_name} RQI")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("RQI")
        axs[i].legend()
        format_axes(axs[i])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "activation_rqi.png"))
    plt.close()
except Exception as e:
    print(f"Error in Activation Function Comparison plots: {e}")
    plt.close()

#################################################################
# 6. Multiple Synthetic Dataset Evaluation - Losses and Metrics
#################################################################
try:
    multi_synth_path = "experiment_results/experiment_a7fae37029a94968b4e62da4dd8e290f_proc_2518847/experiment_data.npy"
    multi_synth_data = np.load(multi_synth_path, allow_pickle=True).item()
    # For this experiment, the ablation_study key holds several distributions: uniform, normal, bimodal.
    distributions_multi = ["uniform", "normal", "bimodal"]
    
    # Loss Curves
    fig, axs = plt.subplots(1, len(distributions_multi), figsize=(5*len(distributions_multi), 4))
    if len(distributions_multi) == 1:
        axs = [axs]
    for i, dist in enumerate(distributions_multi):
        dist_data = multi_synth_data.get("ablation_study", {}).get(dist, {})
        epochs = range(1, len(dist_data.get("losses", {}).get("train", [])) + 1)
        axs[i].plot(epochs, dist_data.get("losses", {}).get("train", []), label="Train Loss", marker="o")
        axs[i].plot(epochs, dist_data.get("losses", {}).get("val", []), label="Validation Loss", marker="o")
        axs[i].set_title(f"{dist.capitalize()} Losses")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        format_axes(axs[i])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multiple_synthetic_losses.png"))
    plt.close()
    
    # Metrics (RQI) for the normal and uniform/bimodal if available
    fig, axs = plt.subplots(1, len(distributions_multi), figsize=(5*len(distributions_multi), 4))
    if len(distributions_multi) == 1:
        axs = [axs]
    for i, dist in enumerate(distributions_multi):
        dist_data = multi_synth_data.get("ablation_study", {}).get(dist, {})
        # Some experiments store metrics under "metrics" at the top-level of the distribution
        # If missing, we skip.
        train_metric = dist_data.get("metrics", {}).get("train", None)
        if train_metric is None:
            axs[i].text(0.5,0.5,"No Metrics", ha="center", va="center")
        else:
            epochs = range(1, len(train_metric) + 1)
            axs[i].plot(epochs, train_metric, label="Train RQI", marker="o")
            axs[i].set_title(f"{dist.capitalize()} Metrics")
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel("RQI")
            axs[i].legend()
        format_axes(axs[i])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multiple_synthetic_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error in Multiple Synthetic Dataset Evaluation plots: {e}")
    plt.close()

##########################################################
# 7. Input Feature Scaling Evaluation - Losses and Predictions
##########################################################
try:
    scaling_path = "experiment_results/experiment_50409cb5bd2945609feb13d85eb4e7f2_proc_2518847/experiment_data.npy"
    scaling_data = np.load(scaling_path, allow_pickle=True).item()
    scaling_eval = scaling_data.get("input_feature_scaling", {})
    scaling_methods = list(scaling_eval.keys())  # e.g., "min_max_scaling", "standardization"
    
    # Loss Curves: Plot training and validation losses for each scaling method on one plot.
    plt.figure(figsize=(8, 6))
    for method in scaling_methods:
        method_data = scaling_eval.get(method, {})
        plt.plot(method_data.get("losses", {}).get("train", []), label=f"Train {method.replace('_',' ')}", marker="o")
        plt.plot(method_data.get("losses", {}).get("val", []), label=f"Validation {method.replace('_',' ')}", linestyle="--", marker="o")
    plt.title("Input Feature Scaling: Training & Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    ax = plt.gca()
    format_axes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "feature_scaling_losses.png"))
    plt.close()
    
    # Predictions vs Ground Truth: Create a subplot for each scaling method.
    fig, axs = plt.subplots(1, len(scaling_methods), figsize=(6*len(scaling_methods), 5))
    if len(scaling_methods) == 1:
        axs = [axs]
    for i, method in enumerate(scaling_methods):
        method_data = scaling_eval.get(method, {})
        gt = method_data.get("ground_truth", [])
        pred = method_data.get("predictions", [])
        axs[i].scatter(gt, pred, alpha=0.5)
        axs[i].plot([min(gt), max(gt)], [min(gt), max(gt)], "r--", label="Ideal")
        axs[i].set_title(f"{method.replace('_',' ').title()} Predictions")
        axs[i].set_xlabel("Ground Truth")
        axs[i].set_ylabel("Predictions")
        axs[i].legend()
        format_axes(axs[i])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "feature_scaling_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error in Input Feature Scaling Evaluation plots: {e}")
    plt.close()

#############################################
# 8. Gradient Clipping Evaluation - Loss Comparison
#############################################
try:
    gc_path = "experiment_results/experiment_11e407aaafa0488b839884e44a22003c_proc_2518847/experiment_data.npy"
    gc_data = np.load(gc_path, allow_pickle=True).item()
    gc_eval = gc_data.get("gradient_clipping", {})
    
    no_clip = gc_eval.get("without_clipping", {})
    with_clip = gc_eval.get("with_clipping", {})
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(no_clip.get("losses", {}).get("train", []), label="Train Loss", marker="o")
    axs[0].plot(no_clip.get("losses", {}).get("val", []), label="Validation Loss", marker="o")
    axs[0].set_title("No Gradient Clipping")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    format_axes(axs[0])
    
    axs[1].plot(with_clip.get("losses", {}).get("train", []), label="Train Loss", marker="o")
    axs[1].plot(with_clip.get("losses", {}).get("val", []), label="Validation Loss", marker="o")
    axs[1].set_title("With Gradient Clipping")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    format_axes(axs[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "gradient_clipping_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error in Gradient Clipping Evaluation plots: {e}")
    plt.close()

##################################################
# 9. Feature Interaction Impact Evaluation - Loss Curves
##################################################
try:
    fi_path = "experiment_results/experiment_95e97459f2964aae973400533fc56ce5_proc_2518845/experiment_data.npy"
    fi_data = np.load(fi_path, allow_pickle=True).item()
    fi_eval = fi_data.get("feature_interaction", {})
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # No Interaction Terms
    axs[0].plot(fi_eval.get("no_interaction", {}).get("losses", {}).get("train", []), label="Train Loss", marker="o")
    axs[0].plot(fi_eval.get("no_interaction", {}).get("losses", {}).get("val", []), label="Validation Loss", marker="o")
    axs[0].set_title("No Interaction Terms")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    format_axes(axs[0])
    # With Interaction Terms
    axs[1].plot(fi_eval.get("with_interaction", {}).get("losses", {}).get("train", []), label="Train Loss", marker="o")
    axs[1].plot(fi_eval.get("with_interaction", {}).get("losses", {}).get("val", []), label="Validation Loss", marker="o")
    axs[1].set_title("With Interaction Terms")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    format_axes(axs[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "feature_interaction_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error in Feature Interaction Impact Evaluation plots: {e}")
    plt.close()

print("All figures have been generated and stored in the 'figures/' directory.")