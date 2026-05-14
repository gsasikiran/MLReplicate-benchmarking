#!/usr/bin/env python3
"""
Final Aggregated Plotting Script for the Prediction vs Capacity Trade-off Paper

This script aggregates the final results from multiple experiments and ablation studies.
Each experiment’s results are loaded from pre‐saved .npy files (using the exact file paths)
and then plotted into scientifically rigorous figures that are saved into the "figures/" folder.
Each figure is produced in a try/except block so that failure in one plot does not affect others.

Sections:
1. Hyperparameter Tuning (Baseline/Research)
2. Multiple Synthetic Datasets Variation
3. Activation Function Variation
4. Regularization Techniques Comparison
5. Dropout Rate Variation
6. Optimizer Variation
7. Batch Size Variation
8. Early Stopping Mechanism
9. Ensemble Model Variation
10. Outlier Impact Assessment
11. Multiple Synthetic Dataset Variation (Predictions & Metrics)
12. Data Distribution Impact

Before running, please ensure that the .npy files exist at the exact paths given.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set the global plot parameters for publication quality.
plt.rcParams.update({'font.size': 14})
DPI = 300

# Create the final figures folder
os.makedirs("figures", exist_ok=True)

# Helper function to remove top/right spines.
def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction='out')

########################################
# 1. Hyperparameter Tuning (Baseline/Research)
########################################
try:
    # Load the experiment data from baseline/research experiment.
    data_path = "experiment_results/experiment_1b52c0484b61485a929a2f7d04b1be97_proc_2538463/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    # Extract hyperparameter tuning data
    hyper_data = exp_data.get("hyperparam_tuning_num_epochs", {}).get("synthetic_dataset", {})
    losses = hyper_data.get("losses", {}).get("train", [])
    accuracy = hyper_data.get("metrics", {}).get("train", [])
    epochs = range(1, len(losses)+1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    axs[0].plot(epochs, losses, marker='o', label="Train Loss")
    axs[0].set_title("Hyperparameter Tuning: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].plot(epochs, accuracy, marker='s', color='orange', label="Train Accuracy")
    axs[1].set_title("Hyperparameter Tuning: Training Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "hyperparameter_tuning.png"))
    plt.close()
except Exception as e:
    print("Error in Hyperparameter Tuning plot:", e)
    plt.close()

########################################
# 2. Multiple Synthetic Datasets Variation (Normal, Uniform, Skewed)
# Combined figure: two rows, three columns (top row: Loss, bottom row: Accuracy)
########################################
try:
    data_path = "experiment_results/experiment_948b31fa0728447e886bd872912f9927_proc_2539011/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    multi_data = exp_data.get("multiple_synthetic_datasets_variation", {})
    dataset_names = ["normal_dataset", "uniform_dataset", "skewed_dataset"]
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), dpi=DPI)
    for i, ds in enumerate(dataset_names):
        ds_data = multi_data.get(ds, {})
        loss = ds_data.get("losses", {}).get("train", [])
        acc = ds_data.get("metrics", {}).get("train", [])
        epochs = range(1, len(loss)+1)
        # Loss plot
        axs[0, i].plot(epochs, loss, marker='o', label="Train Loss")
        axs[0, i].set_title(f"{ds.replace('_', ' ').capitalize()} - Loss")
        axs[0, i].set_xlabel("Epochs")
        axs[0, i].set_ylabel("Loss")
        style_axes(axs[0, i])
        axs[0, i].legend()
        # Accuracy plot
        axs[1, i].plot(epochs, acc, marker='s', color='green', label="Train Accuracy")
        axs[1, i].set_title(f"{ds.replace('_', ' ').capitalize()} - Accuracy")
        axs[1, i].set_xlabel("Epochs")
        axs[1, i].set_ylabel("Accuracy")
        style_axes(axs[1, i])
        axs[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multiple_synthetic_datasets_variation.png"))
    plt.close()
except Exception as e:
    print("Error in Multiple Synthetic Datasets Variation plot:", e)
    plt.close()

########################################
# 3. Activation Function Variation
# Combined figure: 2 subplots (loss and accuracy)
########################################
try:
    data_path = "experiment_results/experiment_4189193744e94f4e906f793e07a65bd1_proc_2539012/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    act_data = exp_data.get("activation_function_variation", {}).get("synthetic_dataset", {})
    loss = act_data.get("losses", [])
    acc = act_data.get("metrics", [])
    epochs = range(1, len(loss)+1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    axs[0].plot(epochs, loss, marker='o', color='purple', label="Train Loss")
    axs[0].set_title("Activation Function Variation: Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].plot(epochs, acc, marker='s', color='brown', label="Train Accuracy")
    axs[1].set_title("Activation Function Variation: Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "activation_function_variation.png"))
    plt.close()
except Exception as e:
    print("Error in Activation Function Variation plot:", e)
    plt.close()

########################################
# 4. Regularization Techniques Comparison (None, L1, L2, Dropout)
# Combined figure: 2 subplots (Loss and Accuracy with four curves each)
########################################
try:
    data_path = "experiment_results/experiment_5fe0fa01789a48c19df9fb2a0e808eb6_proc_2539014/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    reg_keys = ["none", "l1", "l2", "dropout"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    for reg in reg_keys:
        reg_data = exp_data.get(reg, {}).get("synthetic_dataset", {})
        loss = reg_data.get("losses", {}).get("train", [])
        acc = reg_data.get("metrics", {}).get("train", [])
        epochs = range(1, len(loss)+1)
        axs[0].plot(epochs, loss, marker='o', label=f"{reg.upper()}") 
        axs[1].plot(epochs, acc, marker='s', label=f"{reg.upper()}")
    
    axs[0].set_title("Regularization: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].set_title("Regularization: Training Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "regularization_techniques_comparison.png"))
    plt.close()
except Exception as e:
    print("Error in Regularization Techniques Comparison plot:", e)
    plt.close()

########################################
# 5. Dropout Rate Variation
# Combined figure: 2 subplots for Loss and Accuracy over different dropout rates
########################################
try:
    data_path = "experiment_results/experiment_6cc96a93a2134044973af748b594eff7_proc_2539013/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    dropout_data = exp_data.get("dropout_variation", {}).get("synthetic_dataset", {})
    # Assume dropout_data["losses"]["train"] and dropout_data["metrics"]["train"] 
    # are dictionaries with keys as dropout rates ("0.0", "0.2", "0.5")
    loss_dict = dropout_data.get("losses", {}).get("train", {})
    acc_dict = dropout_data.get("metrics", {}).get("train", {})
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    for rate, loss in loss_dict.items():
        epochs = range(1, len(loss)+1)
        axs[0].plot(epochs, loss, marker='o', label=f"Dropout {rate}")
    axs[0].set_title("Dropout Variation: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()

    for rate, acc in acc_dict.items():
        epochs = range(1, len(acc)+1)
        axs[1].plot(epochs, acc, marker='s', label=f"Dropout {rate}")
    axs[1].set_title("Dropout Variation: Training Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "dropout_rate_variation.png"))
    plt.close()
except Exception as e:
    print("Error in Dropout Rate Variation plot:", e)
    plt.close()

########################################
# 6. Optimizer Variation
# Combined figure: 2 subplots (Loss and Accuracy)
########################################
try:
    data_path = "experiment_results/experiment_2ac728ea436a4bfe97e07d4625d77dea_proc_2539012/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    opt_data = exp_data.get("optimizer_variants", {}).get("synthetic_dataset", {})
    loss = opt_data.get("losses", {}).get("train", [])
    acc = opt_data.get("metrics", {}).get("train", [])
    epochs = range(1, len(loss)+1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    axs[0].plot(epochs, loss, marker='o', color='navy', label="Train Loss")
    axs[0].set_title("Optimizer Variation: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].plot(epochs, acc, marker='s', color='teal', label="Train Accuracy")
    axs[1].set_title("Optimizer Variation: Training Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "optimizer_variation.png"))
    plt.close()
except Exception as e:
    print("Error in Optimizer Variation plot:", e)
    plt.close()

########################################
# 7. Batch Size Variation
# Combined figure: 2 subplots (Accuracy and Loss)
########################################
try:
    data_path = "experiment_results/experiment_3d9011406e9940199418a50c6eed3db7_proc_2539014/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    bs_data = exp_data.get("batch_size_variation", {}).get("synthetic_dataset", {})
    acc = bs_data.get("metrics", {}).get("train", [])
    loss = bs_data.get("losses", {}).get("train", [])
    epochs_acc = range(1, len(acc)+1)
    epochs_loss = range(1, len(loss)+1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    axs[0].plot(epochs_acc, acc, marker='o', label="Train Accuracy")
    axs[0].set_title("Batch Size Variation: Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].plot(epochs_loss, loss, marker='x', color='red', label="Train Loss")
    axs[1].set_title("Batch Size Variation: Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "batch_size_variation.png"))
    plt.close()
except Exception as e:
    print("Error in Batch Size Variation plot:", e)
    plt.close()

########################################
# 8. Early Stopping Mechanism
# Combined figure: 2 subplots (Loss Curves and Accuracy Curves) with training and validation curves.
########################################
try:
    data_path = "experiment_results/experiment_4f7f21017ca74dfcbc97309dfc2f36d0_proc_2539014/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    es_data = exp_data.get("early_stopping", {}).get("synthetic_dataset", {})
    train_loss = es_data.get("losses", {}).get("train", [])
    val_loss = es_data.get("losses", {}).get("val", [])
    train_acc = es_data.get("metrics", {}).get("train", [])
    val_acc = es_data.get("metrics", {}).get("val", [])
    epochs_loss = range(1, len(train_loss)+1)
    epochs_acc = range(1, len(train_acc)+1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    axs[0].plot(epochs_loss, train_loss, marker='o', label="Train Loss")
    axs[0].plot(epochs_loss, val_loss, marker='s', label="Validation Loss")
    axs[0].set_title("Early Stopping: Loss Curves")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].plot(epochs_acc, train_acc, marker='o', label="Train Accuracy")
    axs[1].plot(epochs_acc, val_acc, marker='s', label="Validation Accuracy")
    axs[1].set_title("Early Stopping: Accuracy Curves")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "early_stopping_mechanism.png"))
    plt.close()
except Exception as e:
    print("Error in Early Stopping Mechanism plot:", e)
    plt.close()

########################################
# 9. Ensemble Model Variation
# Combined figure: 2 subplots (Loss and Accuracy)
########################################
try:
    data_path = "experiment_results/experiment_25b87341f8234197bdeb302f1b8fad3c_proc_2539012/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    ens_data = exp_data.get("ensemble_model_variation", {}).get("synthetic_dataset", {})
    loss = ens_data.get("losses", {}).get("train", [])
    acc = ens_data.get("metrics", {}).get("train", [])
    epochs = range(1, len(loss)+1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    axs[0].plot(epochs, loss, marker='o', label="Train Loss")
    axs[0].set_title("Ensemble Model: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].plot(epochs, acc, marker='s', color='orange', label="Train Accuracy")
    axs[1].set_title("Ensemble Model: Training Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "ensemble_model_variation.png"))
    plt.close()
except Exception as e:
    print("Error in Ensemble Model Variation plot:", e)
    plt.close()

########################################
# 10. Outlier Impact Assessment
# Combined figure: 2 rows, 3 columns.
# Top row: Training Loss for 0%, 5%, 10% Outliers.
# Bottom row: Training Accuracy for same.
########################################
try:
    data_path = "experiment_results/experiment_f8de536ae8a94c84adceb2722381e6b6_proc_2539012/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    outlier_data = exp_data.get("Outlier Impact Assessment", {})
    outlier_keys = sorted(list(outlier_data.keys()))  # e.g., "0% Outliers", "5% Outliers", "10% Outliers"
    
    fig, axs = plt.subplots(2, len(outlier_keys), figsize=(5*len(outlier_keys), 8), dpi=DPI)
    for i, key in enumerate(outlier_keys):
        data_dict = outlier_data.get(key, {})
        loss = data_dict.get("losses", {}).get("train", [])
        acc = data_dict.get("metrics", {}).get("train", [])
        epochs_loss = range(1, len(loss)+1)
        epochs_acc = range(1, len(acc)+1)
        # Loss subplot (top row)
        axs[0, i].plot(epochs_loss, loss, marker='o', label="Train Loss")
        axs[0, i].set_title(f"{key} - Loss")
        axs[0, i].set_xlabel("Epochs")
        axs[0, i].set_ylabel("Loss")
        style_axes(axs[0, i])
        axs[0, i].legend()
        # Accuracy subplot (bottom row)
        axs[1, i].plot(epochs_acc, acc, marker='s', color='green', label="Train Accuracy")
        axs[1, i].set_title(f"{key} - Accuracy")
        axs[1, i].set_xlabel("Epochs")
        axs[1, i].set_ylabel("Accuracy")
        style_axes(axs[1, i])
        axs[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "outlier_impact_assessment.png"))
    plt.close()
except Exception as e:
    print("Error in Outlier Impact Assessment plot:", e)
    plt.close()

########################################
# 11. Multiple Synthetic Dataset Variation (Ensemble of Predictions & Metrics)
# Two figures: (a) Metrics: side-by-side Loss and Accuracy plots.
# (b) Sample Predictions: Scatter plots for each dataset (Gaussian, Imbalanced, Uniform).
########################################
try:
    data_path = "experiment_results/experiment_d370e5799f234383b0b83f8ab73366b4_proc_2539013/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    multi_syn_data = exp_data.get("multiple_synthetic_datasets", {})
    datasets = ["gaussian", "imbalanced", "uniform"]
    
    # Figure 11a: Metrics plots in one figure with 1 row, 2 columns (Loss and Accuracy - each one combines the three datasets)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    for ds in datasets:
        ds_data = multi_syn_data.get(ds, {})
        loss = ds_data.get("losses", {}).get("train", [])
        acc = ds_data.get("metrics", {}).get("train", [])
        epochs_loss = range(1, len(loss)+1)
        epochs_acc = range(1, len(acc)+1)
        axs[0].plot(epochs_loss, loss, marker='o', label=f"{ds.capitalize()}")
        axs[1].plot(epochs_acc, acc, marker='s', label=f"{ds.capitalize()}")
    
    axs[0].set_title("Multiple Synthetic Dataset Variation: Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].set_title("Multiple Synthetic Dataset Variation: Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multiple_synthetic_dataset_metrics.png"))
    plt.close()
    
    # Figure 11b: Sample Predictions (Scatter plot): Two subplots – one for Ground Truth, one for Predictions for each dataset.
    # We'll combine the three datasets as separate subplots in one row.
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=DPI)
    for i, ds in enumerate(datasets):
        ds_data = multi_syn_data.get(ds, {})
        gt = ds_data.get("ground_truth", [])
        pred = ds_data.get("predictions", [])
        axs[i].scatter(range(len(gt)), gt, color='blue', label="Ground Truth", alpha=0.7)
        axs[i].scatter(range(len(pred)), pred, color='orange', label="Predictions", alpha=0.7)
        axs[i].set_title(f"{ds.capitalize()} Predictions")
        axs[i].set_xlabel("Sample Index")
        axs[i].set_ylabel("Value")
        style_axes(axs[i])
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "multiple_synthetic_dataset_predictions.png"))
    plt.close()
except Exception as e:
    print("Error in Multiple Synthetic Dataset Variation plots:", e)
    plt.close()

########################################
# 12. Data Distribution Impact
# Combined figure: Two subplots: one for Loss curves across distributions and one for Accuracy.
########################################
try:
    data_path = "experiment_results/experiment_e3448d85119548bb96b64746f125ac42_proc_2539014/experiment_data.npy"
    exp_data = np.load(data_path, allow_pickle=True).item()
    ddi_data = exp_data.get("data_distribution_impact", {})
    distributions = list(ddi_data.keys())
    
    # For Loss: one subplot; for Accuracy: another subplot.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
    for dist in distributions:
        dist_data = ddi_data.get(dist, {})
        loss = dist_data.get("losses", {}).get("train", [])
        acc = dist_data.get("metrics", {}).get("train", [])
        epochs_loss = range(1, len(loss)+1)
        epochs_acc = range(1, len(acc)+1)
        axs[0].plot(epochs_loss, loss, marker='o', label=f"{dist.capitalize()}")
        axs[1].plot(epochs_acc, acc, marker='s', label=f"{dist.capitalize()}")
    
    axs[0].set_title("Data Distribution Impact: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    style_axes(axs[0])
    axs[0].legend()
    
    axs[1].set_title("Data Distribution Impact: Training Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    style_axes(axs[1])
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "data_distribution_impact.png"))
    plt.close()
except Exception as e:
    print("Error in Data Distribution Impact plot:", e)
    plt.close()

print("All final plots have been saved in the 'figures/' folder.")