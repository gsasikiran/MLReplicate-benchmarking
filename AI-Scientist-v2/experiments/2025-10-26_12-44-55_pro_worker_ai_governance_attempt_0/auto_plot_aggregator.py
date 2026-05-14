#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16,
    'axes.spines.top': False,
    'axes.spines.right': False
})
os.makedirs("figures", exist_ok=True)

# Load experiment data from provided .npy files
try:
    baseline_data = np.load("experiment_results/experiment_6c7254307104407ba2f45a8dd23e846e_proc_2527448/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading baseline data:", e)
    baseline_data = {}

try:
    multidataset_data = np.load("experiment_results/experiment_bb60ae01b110493d9518fa6e66b5c2d8_proc_2528185/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading multi-dataset data:", e)
    multidataset_data = {}

try:
    networkdepth_data = np.load("experiment_results/experiment_b3e25fd1642b4dd499ce409caf458e90_proc_2528188/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading network depth data:", e)
    networkdepth_data = {}

try:
    activation_data = np.load("experiment_results/experiment_7c16c1a6f4a44d349f2b3731b8a1a3a1_proc_2528187/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading activation data:", e)
    activation_data = {}

try:
    multifeature_data = np.load("experiment_results/experiment_fec8a9cb0d1e4c769ebfc8eb40035fd8_proc_2528185/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading multi-feature data:", e)
    multifeature_data = {}

try:
    traintest_data = np.load("experiment_results/experiment_5606baeeed29435095a9e294d7bad042_proc_2528188/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading train-test data:", e)
    traintest_data = {}

try:
    regularization_data = np.load("experiment_results/experiment_faf656adc8b04c9b9df3b9cffea0b8db_proc_2528186/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading regularization data:", e)
    regularization_data = {}

try:
    datanoise_data = np.load("experiment_results/experiment_1ae4b1ba696346b3853e81f77292b854_proc_2528188/experiment_data.npy", allow_pickle=True).item()
except Exception as e:
    print("Error loading data noise data:", e)
    datanoise_data = {}

# Figure 1: Baseline (Training and Validation Loss with Predictions vs Ground Truth)
try:
    exp = baseline_data.get("hyperparam_tuning_learning_rate", {}).get("synthetic_worker_data", {})
    losses = exp.get("losses", {})
    epochs = range(len(losses.get("train", [])))
    preds = exp.get("predictions", [])
    ground_truth = exp.get("ground_truth", [])
    
    fig, axs = plt.subplots(1,2, figsize=(14,6), dpi=300)
    axs[0].plot(epochs, losses.get("train", []), marker='o', label="Training Loss")
    axs[0].plot(epochs, losses.get("val", []), marker='o', label="Validation Loss")
    axs[0].set_title("Baseline Training and Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    axs[1].scatter(ground_truth, preds, alpha=0.6, label="Predictions")
    if ground_truth:
        min_val, max_val = min(ground_truth), max(ground_truth)
        axs[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    axs[1].set_title("Baseline Predictions vs Ground Truth")
    axs[1].set_xlabel("Ground Truth WIS")
    axs[1].set_ylabel("Predicted WIS")
    axs[1].axis("equal")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure1_Baseline.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 1:", e)
    plt.close()

# Figure 2: Multi-Dataset Evaluation (Aggregated Loss Curves)
try:
    md_eval = multidataset_data.get("multi_dataset_evaluation", {})
    datasets = list(md_eval.keys())
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)
    for ds in datasets:
        data = md_eval.get(ds, {})
        losses = data.get("losses", {})
        eps = range(len(losses.get("train", [])))
        ax.plot(eps, losses.get("train", []), marker='o', label=f"{ds.replace('_',' ').title()} Train")
        ax.plot(eps, losses.get("val", []), marker='o', label=f"{ds.replace('_',' ').title()} Validation")
    ax.set_title("Multi-Dataset Evaluation: Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure2_MultiDataset_Evaluation.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 2:", e)
    plt.close()

# Figure 3: Network Depth Study (Validation Loss and WIS Combined)
try:
    ablation = networkdepth_data.get("ablation_study", {})
    models = ["SimpleNN", "DeepNN3", "DeepNN4"]
    fig, axs = plt.subplots(2, 1, figsize=(10,10), dpi=300)
    for model in models:
        model_data = ablation.get(model, {})
        losses = model_data.get("losses", {})
        eps_loss = range(len(losses.get("val", [])))
        axs[0].plot(eps_loss, losses.get("val", []), marker='o', label=f"{model} Val Loss")
    axs[0].set_title("Network Depth: Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    for model in models:
        model_data = ablation.get(model, {})
        metrics = model_data.get("metrics", {})
        eps_metric = range(len(metrics.get("val", [])))
        axs[1].plot(eps_metric, metrics.get("val", []), marker='o', label=f"{model} WIS")
    axs[1].set_title("Network Depth: Validation WIS")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("WIS")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure3_Network_Depth.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 3:", e)
    plt.close()

# Figure 4: Activation Function Impact (Loss and Predictions Combined)
try:
    act_study = activation_data.get("activation_function_study", {}).get("synthetic_worker_data", {})
    eps = range(1, len(act_study.get("losses", {}).get("train", [])) + 1)
    fig, axs = plt.subplots(1,2, figsize=(14,6), dpi=300)
    axs[0].plot(eps, act_study.get("losses", {}).get("train", []), marker='o', label="Training Loss")
    axs[0].plot(eps, act_study.get("losses", {}).get("val", []), marker='o', label="Validation Loss")
    axs[0].set_title("Activation Study: Loss Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    gt = act_study.get("ground_truth", [])
    preds = act_study.get("predictions", [])
    axs[1].scatter(gt, preds, alpha=0.6, label="Predictions")
    if gt:
        min_val, max_val = min(gt), max(gt)
        axs[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    axs[1].set_title("Activation Study: Predictions vs Ground Truth")
    axs[1].set_xlabel("Ground Truth")
    axs[1].set_ylabel("Predicted Value")
    axs[1].axis("equal")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure4_Activation_Impact.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 4:", e)
    plt.close()

# Figure 5: Multi-Feature Interaction (Loss and WIS in One Figure)
try:
    mf_int = multifeature_data.get("multi_feature_interaction", {})
    fig, axs = plt.subplots(1,2, figsize=(14,6), dpi=300)
    for ds in ["original_dataset", "interaction_dataset"]:
        data = mf_int.get(ds, {})
        losses = data.get("losses", {})
        eps = range(len(losses.get("train", [])))
        axs[0].plot(eps, losses.get("train", []), marker='o', label=f"{ds.replace('_',' ').title()} Train")
        axs[0].plot(eps, losses.get("val", []), marker='o', label=f"{ds.replace('_',' ').title()} Validation")
    axs[0].set_title("Multi-Feature Interaction: Loss Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    for ds in ["original_dataset", "interaction_dataset"]:
        data = mf_int.get(ds, {})
        metrics = data.get("metrics", {})
        eps = range(len(metrics.get("val", [])))
        axs[1].plot(eps, metrics.get("val", []), marker='o', label=f"{ds.replace('_',' ').title()} WIS")
    axs[1].set_title("Multi-Feature Interaction: Validation WIS")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("WIS")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure5_MultiFeature_Interaction.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 5:", e)
    plt.close()

# Figure 6: Train-Test Split Evaluation (Loss and WIS Combined)
try:
    split_data = traintest_data.get("train_test_split_ratios", {})
    splits = list(split_data.keys())
    fig, axs = plt.subplots(2, 1, figsize=(10,8), dpi=300)
    for split in splits:
        data = split_data.get(split, {})
        eps = range(1, len(data.get("losses", {}).get("train", [])) + 1)
        axs[0].plot(eps, data.get("losses", {}).get("train", []), marker='o', label=f"{split.replace('_','/')}" + " Train")
        axs[0].plot(eps, data.get("losses", {}).get("val", []), marker='o', label=f"{split.replace('_','/')}" + " Validation")
    axs[0].set_title("Train-Test Split: Loss Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    for split in splits:
        data = split_data.get(split, {})
        eps = range(1, len(data.get("metrics", {}).get("val", [])) + 1)
        axs[1].plot(eps, data.get("metrics", {}).get("val", []), marker='o', label=f"{split.replace('_','/')}" + " WIS")
    axs[1].set_title("Train-Test Split: Validation WIS")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("WIS")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure6_TrainTest_Split.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 6:", e)
    plt.close()

# Figure 7: Regularization Techniques (Validation Loss and Predictions Combined)
try:
    reg_types = ["regularization_no_regularization", "regularization_l2", "regularization_dropout"]
    dataset = "synthetic_worker_data"
    fig, axs = plt.subplots(1,2, figsize=(14,6), dpi=300)
    for reg in reg_types:
        dat = regularization_data.get(reg, {}).get(dataset, {})
        eps = range(len(dat.get("losses", {}).get("train", [])))
        axs[0].plot(eps, dat.get("losses", {}).get("val", []), marker='o', label=reg.replace('_',' ').title())
    axs[0].set_title("Regularization: Validation Loss Comparison")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    for reg in reg_types:
        dat = regularization_data.get(reg, {}).get(dataset, {})
        axs[1].scatter(dat.get("ground_truth", []), dat.get("predictions", []), alpha=0.6, label=reg.replace('_',' ').title())
    sample_gt = regularization_data.get(reg_types[0], {}).get(dataset, {}).get("ground_truth", [])
    if sample_gt:
        min_val, max_val = min(sample_gt), max(sample_gt)
        axs[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    axs[1].set_title("Regularization: Predictions vs Ground Truth")
    axs[1].set_xlabel("Ground Truth")
    axs[1].set_ylabel("Predicted Value")
    axs[1].axis("equal")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure7_Regularization.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 7:", e)
    plt.close()

# Figure 8: Data Noise Impact (Loss and Predictions Combined in 2x1 Subplots)
try:
    noise_levels = ["no_noise", "low_noise", "medium_noise", "high_noise"]
    fig, axs = plt.subplots(2, 1, figsize=(10,12), dpi=300)
    # Aggregate losses for all noise levels in one subplot
    for noise in noise_levels:
        data = datanoise_data.get("data_noise_impact", {}).get(noise, {})
        eps = range(len(data.get("losses", {}).get("train", [])))
        axs[0].plot(eps, data.get("losses", {}).get("train", []), marker='o', label=f"{noise.replace('_',' ').title()} Train")
        axs[0].plot(eps, data.get("losses", {}).get("val", []), marker='o', label=f"{noise.replace('_',' ').title()} Validation")
    axs[0].set_title("Data Noise Impact: Loss Curves")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    # Aggregate predictions for all noise levels in the second subplot (using markers)
    for noise in noise_levels:
        data = datanoise_data.get("data_noise_impact", {}).get(noise, {})
        gt = data.get("ground_truth", [])
        preds = data.get("predictions", [])
        axs[1].scatter(gt, preds, alpha=0.6, label=f"{noise.replace('_',' ').title()}")
        if gt:
            min_val, max_val = min(gt), max(gt)
            axs[1].plot([min_val, max_val], [min_val, max_val], "r--")
    axs[1].set_title("Data Noise Impact: Predictions vs Ground Truth")
    axs[1].set_xlabel("Ground Truth")
    axs[1].set_ylabel("Predicted Value")
    axs[1].axis("equal")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure8_Data_Noise_Impact.png"))
    plt.close()
except Exception as e:
    print("Error in Figure 8:", e)
    plt.close()

print("Final aggregated plots saved in the 'figures' directory.")