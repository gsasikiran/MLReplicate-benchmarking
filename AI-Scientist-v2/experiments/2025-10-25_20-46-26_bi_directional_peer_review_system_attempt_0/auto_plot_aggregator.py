#!/usr/bin/env python3
"""
Final Aggregator Script for Bi-Directional Peer Review Experiments

This script loads pre‐computed experiment results (stored as .npy files)
from various experimental runs and generates a comprehensive set of 
final scientific plots. All plots are saved in the "figures" directory.
Each plotting call is wrapped in a try-except block so that errors in one 
plot do not prevent the full execution of the script.

Before running, please ensure that all the .npy files exist at the full and
exact file paths provided in the experiment summaries.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Use larger font sizes for final publication
plt.rcParams.update({'font.size': 14})

# Create the output directory for final figures
os.makedirs("figures", exist_ok=True)


# -----------------------------
# 1. Activation Function Tuning Loss Curves (Baseline/Research)
# -----------------------------
try:
    baseline_data = np.load("experiment_results/experiment_b295ac7cb7eb4d678ea382df2460d5d4_proc_2514469/experiment_data.npy", allow_pickle=True).item()
    # Expected structure: 
    # { "activation_function_tuning": { "FeedbackDataset": { "losses": { act_func: {"train": [...], "val": [...]}, ... }, ... } } }
    af_tuning = baseline_data.get("activation_function_tuning", {}).get("FeedbackDataset", {}).get("losses", {})
    act_funcs = list(af_tuning.keys())
    n_funcs = len(act_funcs)
    if n_funcs == 0:
        raise ValueError("No activation function loss data found.")
    fig, axs = plt.subplots(1, n_funcs, figsize=(5 * n_funcs, 4))
    if n_funcs == 1:
        axs = [axs]
    for i, func in enumerate(act_funcs):
        train_losses = af_tuning[func].get("train", [])
        val_losses = af_tuning[func].get("val", [])
        epochs = list(range(1, len(train_losses) + 1))
        axs[i].plot(epochs, train_losses, label="Training Loss")
        axs[i].plot(epochs, val_losses, label="Validation Loss")
        axs[i].set_title(f"Loss Curves: {func}")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "activation_function_tuning_loss_curves.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 1 (Activation Function Tuning Loss Curves):", e)


# -----------------------------
# 2. Feedback Dataset Training Metrics (Baseline/Research)
# -----------------------------
try:
    train_metrics = baseline_data.get("activation_function_tuning", {}).get("FeedbackDataset", {}).get("metrics", {}).get("train", [])
    if not train_metrics:
        raise ValueError("No training metrics found.")
    epochs = list(range(1, len(train_metrics) + 1))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(epochs, train_metrics, label="Training Metrics")
    ax.set_title("Training Metrics Over Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "training_metrics.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 2 (Training Metrics):", e)


# -----------------------------
# 3. Multiple Synthetic Datasets Loss Curves (Ablation Study)
# -----------------------------
try:
    ablation_data = np.load("experiment_results/experiment_ca8e3ba84cf949488aca90fea6196495_proc_2515358/experiment_data.npy", allow_pickle=True).item()
    msyn = ablation_data.get("multiple_synthetic_datasets", {})
    datasets = list(msyn.keys())
    if not datasets:
        raise ValueError("No multiple synthetic dataset data found.")
    n_ds = len(datasets)
    fig, axs = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4))
    if n_ds == 1:
        axs = [axs]
    for i, ds in enumerate(datasets):
        losses = msyn[ds].get("losses", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        epochs = list(range(1, len(train_loss) + 1))
        axs[i].plot(epochs, train_loss, label="Training Loss")
        axs[i].plot(epochs, val_loss, label="Validation Loss")
        axs[i].set_title(f"{ds} Loss Curves")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "multiple_synthetic_datasets_loss_curves.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 3 (Multiple Synthetic Datasets Loss Curves):", e)


# -----------------------------
# 4. Feature Scaling Ablation: Scaled vs Unscaled Data
# -----------------------------
try:
    scaling_data = np.load("experiment_results/experiment_7138c462abc045f6afe842a44187bad2_proc_2515359/experiment_data.npy", allow_pickle=True).item()
    fs_ablation = scaling_data.get("feature_scaling_ablation", {})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for idx, key in enumerate(["scaled_data", "unscaled_data"]):
        ds = fs_ablation.get(key, {})
        losses = ds.get("losses", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        epochs = list(range(1, len(train_loss) + 1))
        axs[idx].plot(epochs, train_loss, label="Training Loss")
        axs[idx].plot(epochs, val_loss, label="Validation Loss")
        title = key.replace("_", " ").capitalize()
        axs[idx].set_title(f"{title} Losses")
        axs[idx].set_xlabel("Epochs")
        axs[idx].set_ylabel("Loss")
        axs[idx].legend()
        axs[idx].spines["top"].set_visible(False)
        axs[idx].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "feature_scaling_ablation.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 4 (Feature Scaling Ablation):", e)


# -----------------------------
# 5. Dropout Regularization: Loss Curves and Training Metrics
# -----------------------------
try:
    dropout_data = np.load("experiment_results/experiment_f75ad11b9898486b98f215c2dc2dcb7a_proc_2515359/experiment_data.npy", allow_pickle=True).item()
    dropout_ds = dropout_data.get("dropout_ablation", {}).get("FeedbackDataset", {})
    loss_info = dropout_ds.get("losses", {})
    metric_info = dropout_ds.get("metrics", {})
    epochs_loss = list(range(1, len(loss_info.get("train", [])) + 1))
    epochs_metric = list(range(1, len(metric_info.get("train", [])) + 1))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(epochs_loss, loss_info.get("train", []), label="Training Loss")
    axs[0].plot(epochs_loss, loss_info.get("val", []), label="Validation Loss")
    axs[0].set_title("Dropout Loss Curves")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].plot(epochs_metric, metric_info.get("train", []), label="Training Metric")
    axs[1].set_title("Dropout Training Metrics")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Metric Value")
    axs[1].legend()
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "dropout_regularization.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 5 (Dropout Regularization):", e)


# -----------------------------
# 6. Variability of Input Features Loss Curves
# -----------------------------
try:
    varfeat_data = np.load("experiment_results/experiment_b202bcba75d14e1d8571d0e2d56d1653_proc_2515361/experiment_data.npy", allow_pickle=True).item()
    var_ds = varfeat_data.get("variability_of_input_features", {}).get("FeedbackDataset", {})
    losses = var_ds.get("losses", {})
    epochs = list(range(1, len(losses.get("train", [])) + 1))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(epochs, losses.get("train", []), label="Training Loss")
    ax.plot(epochs, losses.get("val", []), label="Validation Loss")
    ax.set_title("Variability of Input Features Loss Curves")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "variability_input_features.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 6 (Variability of Input Features):", e)


# -----------------------------
# 7. Activation Function Variability Across Layers (Train vs Validation)
# -----------------------------
try:
    actvar_data = np.load("experiment_results/experiment_e839e7c3bba54a84bab98d5cee8e4177_proc_2515361/experiment_data.npy", allow_pickle=True).item()
    actvar_ds = actvar_data.get("activation_function_variability", {}).get("FeedbackDataset", {})
    losses = actvar_ds.get("losses", {})
    epochs = list(range(1, len(losses.get("train", [])) + 1))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(epochs, losses.get("train", []), label="Training Loss")
    axs[0].set_title("Activation Variability Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].plot(epochs, losses.get("val", []), label="Validation Loss", color="orange")
    axs[1].set_title("Activation Variability Validation Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "activation_function_variability.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 7 (Activation Function Variability):", e)
    

# -----------------------------
# 8. Multi-Dataset Evaluation (Loss Curves for Default, Alternate1, Alternate2)
# -----------------------------
try:
    multidata = np.load("experiment_results/experiment_2cabfcedd90e43cc8eedb16b2fdabe5f_proc_2515358/experiment_data.npy", allow_pickle=True).item()
    multi_eval = multidata.get("multi_dataset_evaluation", {})
    dataset_list = ["default", "alternate1", "alternate2"]
    fig, axs = plt.subplots(1, len(dataset_list), figsize=(5 * len(dataset_list), 4))
    if len(dataset_list) == 1:
        axs = [axs]
    for i, ds in enumerate(dataset_list):
        ds_data = multi_eval.get(ds, {})
        losses = ds_data.get("losses", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        epochs = list(range(1, len(train_loss) + 1))
        axs[i].plot(epochs, train_loss, label="Training Loss")
        axs[i].plot(epochs, val_loss, label="Validation Loss")
        axs[i].set_title(f"{ds.capitalize()} Dataset Losses")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "multi_dataset_evaluation.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 8 (Multi-Dataset Evaluation):", e)


# -----------------------------
# 9. Use of Batch Normalization (Loss Curves)
# -----------------------------
try:
    bn_data = np.load("experiment_results/experiment_612c1c4b759740b6ac95452a578b5aa0_proc_2515359/experiment_data.npy", allow_pickle=True).item()
    bn_ds = bn_data.get("batch_normalization", {}).get("FeedbackDataset", {})
    losses = bn_ds.get("losses", {})
    epochs = list(range(1, len(losses.get("train", [])) + 1))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(epochs, losses.get("train", []), label="Training Loss")
    ax.plot(epochs, losses.get("val", []), label="Validation Loss")
    ax.set_title("Batch Normalization Loss Curves")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "batch_normalization_loss.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 9 (Batch Normalization):", e)


# -----------------------------
# 10. Dataset Size Variation (Loss Curves for Different Sizes)
# -----------------------------
try:
    dsize_data = np.load("experiment_results/experiment_2c2004a41ea748dcbf76064b28d5ab85_proc_2515361/experiment_data.npy", allow_pickle=True).item()
    size_var = dsize_data.get("dataset_size_variation", {})
    size_keys = list(size_var.keys())
    if not size_keys:
        raise ValueError("No dataset size variation data found.")
    n_sizes = len(size_keys)
    fig, axs = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 4))
    if n_sizes == 1:
        axs = [axs]
    for i, key in enumerate(size_keys):
        dataset = size_var.get(key, {})
        losses = dataset.get("losses", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        epochs = list(range(1, len(train_loss) + 1))
        axs[i].plot(epochs, train_loss, label="Training Loss")
        axs[i].plot(epochs, val_loss, label="Validation Loss")
        axs[i].set_title(f"Loss Curves for {key}")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "dataset_size_variation.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 10 (Dataset Size Variation):", e)


# -----------------------------
# 11. Multiple Synthetic Dataset Evaluation (One Example)
# -----------------------------
try:
    msde_data = np.load("experiment_results/experiment_554154d05a004ac9b41d7cd1e53ed9df_proc_2515358/experiment_data.npy", allow_pickle=True).item()
    msde_eval = msde_data.get("MultipleSyntheticDatasetEvaluation", {})
    msde_keys = list(msde_eval.keys())
    if not msde_keys:
        raise ValueError("No multiple synthetic dataset evaluation data found.")
    n_msde = len(msde_keys)
    fig, axs = plt.subplots(1, n_msde, figsize=(5 * n_msde, 4))
    if n_msde == 1:
        axs = [axs]
    for i, key in enumerate(msde_keys):
        losses = msde_eval.get(key, {}).get("losses", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        epochs = list(range(1, len(train_loss) + 1))
        axs[i].plot(epochs, train_loss, label="Training Loss")
        axs[i].plot(epochs, val_loss, label="Validation Loss")
        axs[i].set_title(f"{key} Loss Curves")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "multiple_synthetic_dataset_evaluation.png"), dpi=300)
    plt.close(fig)
except Exception as e:
    print("Error in Plot 11 (Multiple Synthetic Dataset Evaluation):", e)


# -----------------------------
# 12. Activation Function Depth Variation:
#     (a) Loss Curves across different activation function combinations
#     (b) Predictions vs Ground Truth for one selected combination
# -----------------------------
try:
    depth_data = np.load("experiment_results/experiment_5a4de164f8bb48c28d40384882476c86_proc_2515361/experiment_data.npy", allow_pickle=True).item()
    act_depth = depth_data.get("activation_depth_variation", {})
    depth_keys = list(act_depth.keys())
    if not depth_keys:
        raise ValueError("No activation function depth variation data found.")
    # (a) Loss Curves for each depth variation combination
    fig1, axs1 = plt.subplots(1, len(depth_keys), figsize=(5 * len(depth_keys), 4))
    if len(depth_keys) == 1:
        axs1 = [axs1]
    for i, key in enumerate(depth_keys):
        losses = act_depth.get(key, {}).get("losses", {})
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        epochs = list(range(1, len(train_loss) + 1))
        axs1[i].plot(epochs, train_loss, label="Training Loss")
        axs1[i].plot(epochs, val_loss, label="Validation Loss")
        axs1[i].set_title(f"{key} Loss Curves")
        axs1[i].set_xlabel("Epochs")
        axs1[i].set_ylabel("Loss")
        axs1[i].legend()
        axs1[i].spines["top"].set_visible(False)
        axs1[i].spines["right"].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join("figures", "activation_depth_variation_loss_curves.png"), dpi=300)
    plt.close(fig1)
    
    # (b) Predictions vs Ground Truth (aggregate from first available combination)
    if depth_keys:
        key0 = depth_keys[0]
        pred_list = act_depth.get(key0, {}).get("predictions", [])
        gt_list = act_depth.get(key0, {}).get("ground_truth", [])
        if pred_list and gt_list:
            preds = np.concatenate(pred_list)
            gt = np.concatenate(gt_list)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.scatter(gt, preds, alpha=0.7)
            ax2.set_title(f"{key0} Predictions vs Ground Truth")
            ax2.set_xlabel("Ground Truth")
            ax2.set_ylabel("Predictions")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            fig2.tight_layout()
            fig2.savefig(os.path.join("figures", "activation_depth_variation_predictions_vs_gt.png"), dpi=300)
            plt.close(fig2)
except Exception as e:
    print("Error in Plot 12 (Activation Function Depth Variation):", e)
    
print("Final figures have been generated and saved in the 'figures/' directory.")