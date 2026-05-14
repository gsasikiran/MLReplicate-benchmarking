#!/usr/bin/env python3
"""
Final Aggregator Script for Adaptive Bayesian Conformal Prediction Experiments
This script loads experiment results from existing .npy files and produces the final set of figures.
All figures are saved in the "figures/" directory.
Each figure is generated in its own try-except block.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set a larger font size for publication quality
plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})
os.makedirs("figures", exist_ok=True)

# --------------------- Figure 1 ---------------------
# Figure 1: Baseline Synthetic Data Loss Curves (Training vs Validation)
try:
    # Load baseline experiment data
    baseline_file = "experiment_results/experiment_ca0dd885309e4e84a53b8dafee5b6d39_proc_2543538/experiment_data.npy"
    data_baseline = np.load(baseline_file, allow_pickle=True).item()
    losses = data_baseline["hyperparam_tuning_momentum"]["synthetic_data"]["losses"]
    epochs = range(1, len(losses["train"]) + 1)
    
    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(epochs, losses["train"], label="Training Loss", marker="o")
    plt.plot(epochs, losses["val"], label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Baseline Synthetic Data: Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure1_Baseline_Synthetic_Loss_Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 1: {e}")
    plt.close()

# --------------------- Figure 2 ---------------------
# Figure 2: Baseline Synthetic Data Predictions vs Ground Truth
try:
    predictions = np.array(data_baseline["hyperparam_tuning_momentum"]["synthetic_data"]["predictions"]).mean(axis=0)
    ground_truth = np.array(data_baseline["hyperparam_tuning_momentum"]["synthetic_data"]["ground_truth"]).mean(axis=0)
    
    plt.figure(figsize=(8,6), dpi=300)
    plt.scatter(ground_truth, predictions, alpha=0.7, label="Predictions")
    plt.plot([ground_truth.min(), ground_truth.max()],
             [ground_truth.min(), ground_truth.max()],
             "r--", label="Ideal")
    plt.xlabel("Ground Truth")
    plt.ylabel("Model Predictions")
    plt.title("Baseline Synthetic Data: Predictions vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure2_Baseline_Predictions_vs_GroundTruth.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 2: {e}")
    plt.close()

# --------------------- Figure 3 ---------------------
# Figure 3: Multiple Synthetic Datasets Evaluation (Linear, Quadratic, Cubic Loss Curves)
try:
    ablation_file1 = "experiment_results/experiment_5cae44f15e6444bbaccd647091b68bca_proc_2544330/experiment_data.npy"
    data_ablation1 = np.load(ablation_file1, allow_pickle=True).item()
    datasets = ["linear_dataset", "quadratic_dataset", "cubic_dataset"]
    
    fig, axs = plt.subplots(1, 3, figsize=(18,6), dpi=300)
    for i, key in enumerate(datasets):
        loss_data = data_ablation1["multiple_datasets_evaluation"][key]["losses"]
        epochs = range(1, len(loss_data["train"]) + 1)
        axs[i].plot(epochs, loss_data["train"], label="Training Loss", marker="o")
        axs[i].plot(epochs, loss_data["val"], label="Validation Loss", marker="s")
        axs[i].set_title(key.replace("_", " ").capitalize())
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure3_MultiSyntheticDatasets_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 3: {e}")
    plt.close()

# --------------------- Figure 4 ---------------------
# Figure 4: Comparison of Baseline vs Learning Rate Scheduler Loss Curves
try:
    ablation_file2 = "experiment_results/experiment_2b4882a4237d4c958205520b15fa1b73_proc_2544331/experiment_data.npy"
    data_ablation2 = np.load(ablation_file2, allow_pickle=True).item()
    
    # Get losses for baseline and with scheduler
    loss_baseline = data_ablation2["baseline"]["synthetic_data"]["losses"]
    loss_scheduler = data_ablation2["ablation_learning_rate_scheduler"]["synthetic_data"]["losses"]
    epochs_base = range(1, len(loss_baseline["train"]) + 1)
    epochs_sched = range(1, len(loss_scheduler["train"]) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(16,6), dpi=300)
    axs[0].plot(epochs_base, loss_baseline["train"], label="Train Loss", marker="o")
    axs[0].plot(epochs_base, loss_baseline["val"], label="Validation Loss", marker="s")
    axs[0].set_title("Baseline (No Scheduler)")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    axs[1].plot(epochs_sched, loss_scheduler["train"], label="Train Loss", marker="o")
    axs[1].plot(epochs_sched, loss_scheduler["val"], label="Validation Loss", marker="s")
    axs[1].set_title("With Learning Rate Scheduler")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure4_LRScheduler_Comparison_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 4: {e}")
    plt.close()

# --------------------- Figure 5 ---------------------
# Figure 5: Input Feature Dimensionality Reduction Loss Curves (1D, 3D, 5D)
try:
    reduction_file = "experiment_results/experiment_134e8a487b85450297e967611b9cd214_proc_2544333/experiment_data.npy"
    data_reduction = np.load(reduction_file, allow_pickle=True).item()
    dims = ["1D", "3D", "5D"]
    
    fig, axs = plt.subplots(1, 3, figsize=(18,6), dpi=300)
    for i, dim in enumerate(dims):
        loss_data = data_reduction["input_feature_dimensionality_reduction"][dim]["losses"]
        epochs = range(1, len(loss_data["train"]) + 1)
        axs[i].plot(epochs, loss_data["train"], label="Training Loss", marker="o", color="blue")
        axs[i].plot(epochs, loss_data["val"], label="Validation Loss", marker="s", color="orange")
        axs[i].set_title(f"{dim} Reduction")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure5_InputFeatureDimensionality_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 5: {e}")
    plt.close()

# --------------------- Figure 6 ---------------------
# Figure 6: Weight Initialization Ablation - Loss Curves for Uniform, Xavier, He
try:
    weightinit_file = "experiment_results/experiment_d6d96f0643e844d6984beea275c65f5e_proc_2544331/experiment_data.npy"
    data_weightinit = np.load(weightinit_file, allow_pickle=True).item()
    methods = ["uniform", "xavier", "he"]
    
    fig, axs = plt.subplots(1, 3, figsize=(18,6), dpi=300)
    for i, method in enumerate(methods):
        loss_data = data_weightinit["weight_initialization_ablation"][method]["losses"]
        epochs = range(1, len(loss_data["train"]) + 1)
        axs[i].plot(epochs, loss_data["train"], label="Training Loss", marker="o")
        axs[i].plot(epochs, loss_data["val"], label="Validation Loss", marker="s")
        axs[i].set_title(f"{method.capitalize()} Initialization")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure6_WeightInitialization_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 6: {e}")
    plt.close()

# --------------------- Figure 7 ---------------------
# Figure 7: Activation Function Exploration - Loss Curves for ReLU, Leaky ReLU, Sigmoid
try:
    activation_file = "experiment_results/experiment_8cda0c51c61c43a5a2175ae58d0aaeee_proc_2544330/experiment_data.npy"
    data_activation = np.load(activation_file, allow_pickle=True).item()
    activations = list(data_activation["activation_function_exploration"].keys())
    
    fig, axs = plt.subplots(1, 3, figsize=(18,6), dpi=300)
    for i, act in enumerate(activations):
        loss_data = data_activation["activation_function_exploration"][act]["losses"]
        epochs = range(1, len(loss_data["train"]) + 1)
        axs[i].plot(epochs, loss_data["train"], label="Training Loss", marker="o")
        axs[i].plot(epochs, loss_data["val"], label="Validation Loss", marker="s")
        axs[i].set_title(f"{act}")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure7_ActivationFunctions_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 7: {e}")
    plt.close()

# --------------------- Figure 8 ---------------------
# Figure 8: Optimizer Comparison - Scatter Plots of Predictions vs Ground Truth for 4 Optimizers
try:
    optimizer_file = "experiment_results/experiment_6ec5bef52c76482d98dd0b1006f95f3e_proc_2544331/experiment_data.npy"
    data_optimizer = np.load(optimizer_file, allow_pickle=True).item()
    optimizers = list(data_optimizer["optimizer_comparison"].keys())
    
    fig, axs = plt.subplots(2, 2, figsize=(14,12), dpi=300)
    for i, opt in enumerate(optimizers):
        opt_data = data_optimizer["optimizer_comparison"][opt]
        gt = np.concatenate(opt_data["ground_truth"])
        preds = np.concatenate(opt_data["predictions"])
        ax = axs[i // 2, i % 2]
        ax.scatter(gt, preds, alpha=0.5)
        ax.plot([gt.min(), gt.max()], [gt.min(), gt.max()], "r--")
        ax.set_title(f"{opt} Predictions vs Ground Truth")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.axis("equal")
        ax.legend([opt])
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure8_Optimizers_PredictionsScatter.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 8: {e}")
    plt.close()

# --------------------- Figure 9 ---------------------
# Figure 9: Multiple Synthetic Datasets Evaluation (Variance in Noise Levels) - Loss Curves
try:
    noise_file = "experiment_results/experiment_e8830879ece04a65bc22cd6df8408630_proc_2544330/experiment_data.npy"
    data_noise = np.load(noise_file, allow_pickle=True).item()
    noise_dict = data_noise["multiple_synthetic_datasets"]
    
    # Determine number of noise types and create subplots
    keys = list(noise_dict.keys())
    fig, axs = plt.subplots(1, len(keys), figsize=(6*len(keys),6), dpi=300)
    if len(keys) == 1:
        axs = [axs]
    for i, noise in enumerate(keys):
        loss_data = noise_dict[noise]["losses"]
        epochs = range(1, len(loss_data["train"]) + 1)
        axs[i].plot(epochs, loss_data["train"], label="Training Loss", marker="o")
        axs[i].plot(epochs, loss_data["val"], label="Validation Loss", marker="s")
        axs[i].set_title(f"{noise.capitalize()} Noise")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure9_NoiseLevels_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 9: {e}")
    plt.close()

# --------------------- Figure 10 ---------------------
# Figure 10: Multiple Synthetic Datasets with Varying Noise Distributions - Loss Curves Comparison
try:
    varied_noise_file = "experiment_results/experiment_48962f80d0124b44b5ec6f8d1bd324dc_proc_2544330/experiment_data.npy"
    data_varied_noise = np.load(varied_noise_file, allow_pickle=True).item()
    # Expect keys: e.g., "Gaussian_High_Noise", "Uniform_Noise", "Gaussian_Low_Noise"
    noise_keys = list(data_varied_noise.keys())
    
    fig, axs = plt.subplots(1, len(noise_keys), figsize=(6*len(noise_keys),6), dpi=300)
    if len(noise_keys)==1:
        axs = [axs]
    for i, nk in enumerate(noise_keys):
        loss_data = data_varied_noise[nk]["losses"]
        epochs = range(1, len(loss_data["train"]) + 1)
        axs[i].plot(epochs, loss_data["train"], label="Training Loss", marker="o")
        axs[i].plot(epochs, loss_data["val"], label="Validation Loss", marker="s")
        axs[i].set_title(nk.replace("_", " "))
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure10_VariedNoise_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 10: {e}")
    plt.close()

# --------------------- Figure 11 ---------------------
# Figure 11: Multiple Synthetic Datasets with Varying Noise Distributions - Reliability Measures
try:
    noise_keys = list(data_varied_noise.keys())
    fig, axs = plt.subplots(1, len(noise_keys), figsize=(6*len(noise_keys),6), dpi=300)
    if len(noise_keys)==1:
        axs = [axs]
    for i, nk in enumerate(noise_keys):
        try:
            reliability = data_varied_noise[nk]["metrics"]["val"]
            epochs = range(1, len(reliability) + 1)
            axs[i].plot(epochs, reliability, label="Reliability Measure", marker="o")
            axs[i].set_title(f"{nk.replace('_', ' ')} Reliability")
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel("Reliability")
            axs[i].legend()
        except Exception as inner_e:
            axs[i].text(0.5, 0.5, f"No reliability data for {nk}", horizontalalignment="center")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure11_VariedNoise_Reliability.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 11: {e}")
    plt.close()

# --------------------- Figure 12 ---------------------
# Figure 12: Impact of Input Feature Scaling Methods - Loss Curves for Different Scaling Types
try:
    scaling_file = "experiment_results/experiment_a42b4bdd4e1f48ba997762121ea9c143_proc_2544331/experiment_data.npy"
    data_scaling = np.load(scaling_file, allow_pickle=True).item()
    scaling_methods = list(data_scaling["feature_scaling"].keys())
    
    fig, axs = plt.subplots(1, len(scaling_methods), figsize=(6*len(scaling_methods),6), dpi=300)
    if len(scaling_methods) == 1:
        axs = [axs]
    for i, method in enumerate(scaling_methods):
        loss_data = data_scaling["feature_scaling"][method]["losses"]
        epochs = range(1, len(loss_data["train"]) + 1)
        axs[i].plot(epochs, loss_data["train"], label="Training Loss", marker="o")
        axs[i].plot(epochs, loss_data["val"], label="Validation Loss", marker="s")
        axs[i].set_title(f"{method.capitalize()} Scaling")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "Figure12_FeatureScaling_LossCurves.png"))
    plt.close()
except Exception as e:
    print(f"Error in Figure 12: {e}")
    plt.close()

print("All final figures have been generated and saved in the 'figures/' directory.")