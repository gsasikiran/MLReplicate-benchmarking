#!/usr/bin/env python3
"""
Final Aggregated Plot Script for COLLAB LLM Experiments
This script loads the published .npy experiment data from various experiment summaries,
aggregates the findings into final, publication‐grade figures, and saves them in the
"figures/" directory. Each figure is created in its own try‐except block so that one
failure does not stop the rest of the plots from being produced.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Global style settings for publication quality
plt.rcParams.update({'font.size': 14})
dpi = 300

def set_spines(ax):
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Create directory for final figures.
os.makedirs("figures", exist_ok=True)


# ------------------------- Figure 1: Baseline Results -------------------------
# Aggregated plot from the baseline experiment (training loss and UES over epochs).
try:
    baseline_file = "experiment_results/experiment_12d9b52265dc47afaec05d2e6972d4da_proc_2539767/experiment_data.npy"
    baseline_data = np.load(baseline_file, allow_pickle=True).item()
    loss_baseline = baseline_data["momentum_tuning"]["synthetic_dataset"]["losses"]["train"]
    ues_baseline = baseline_data["momentum_tuning"]["synthetic_dataset"]["metrics"]["train"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    # Plot training loss
    axs[0].plot(loss_baseline, label="Training Loss")
    axs[0].set_title("Baseline Training Loss Over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    set_spines(axs[0])
    # Plot UES metric
    axs[1].plot(ues_baseline, label="User Engagement Score", color="orange")
    axs[1].set_title("Baseline UES Over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("UES")
    axs[1].legend()
    set_spines(axs[1])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig1_baseline.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 1 (Baseline):", e)


# ------------------------- Figure 2: Multi-Dataset Performance Evaluation (Loss) -------------------------
# Using the multi-dataset evaluation experiment; we plot training loss
try:
    file2 = "experiment_results/experiment_6aab4337790743c2ae6e6aebb093e60c_proc_2540411/experiment_data.npy"
    data2 = np.load(file2, allow_pickle=True).item()
    multi_eval = data2["multi_dataset_evaluation"]
    # Use sorted keys for consistency (these may be dataset_1, dataset_2, dataset_3)
    keys2 = sorted(multi_eval.keys())
    n2 = len(keys2)
    fig, axs = plt.subplots(1, n2, figsize=(4 * n2, 4), dpi=dpi)
    if n2 == 1:
        axs = [axs]
    for i, key in enumerate(keys2):
        loss_curve = multi_eval[key]["losses"]["train"]
        axs[i].plot(loss_curve, label="Loss")
        axs[i].set_title(f"{key} Training Loss")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig2_multi_dataset_loss.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 2 (Multi-Dataset Loss):", e)


# ------------------------- Figure 3: Multi-Dataset Performance Evaluation (UES) -------------------------
# Plot UES metrics from the multi-dataset evaluation experiment.
try:
    fig, axs = plt.subplots(1, n2, figsize=(4 * n2, 4), dpi=dpi)
    if n2 == 1:
        axs = [axs]
    for i, key in enumerate(keys2):
        ues_curve = multi_eval[key]["metrics"]["train"]
        axs[i].plot(ues_curve, label="UES", color="orange")
        axs[i].set_title(f"{key} UES")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("UES")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig3_multi_dataset_ues.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 3 (Multi-Dataset UES):", e)


# ------------------------- Figure 4: Input Feature Variability Analysis (Loss) -------------------------
# From the input feature variability experiment, plot training loss for each feature dimension.
try:
    file4 = "experiment_results/experiment_8b687a4be3714db883a6957661088f10_proc_2540413/experiment_data.npy"
    data4 = np.load(file4, allow_pickle=True).item()
    feat_var = data4["input_feature_variability_analysis"]
    keys4 = sorted(feat_var.keys())  # e.g., dataset_dim_5, dataset_dim_10, dataset_dim_15
    n4 = len(keys4)
    fig, axs = plt.subplots(1, n4, figsize=(4 * n4, 4), dpi=dpi)
    if n4 == 1:
        axs = [axs]
    for i, key in enumerate(keys4):
        loss_curve = feat_var[key]["losses"]["train"]
        axs[i].plot(loss_curve, label="Loss")
        axs[i].set_title(f"{key} Training Loss")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig4_feature_variability_loss.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 4 (Feature Variability Loss):", e)


# ------------------------- Figure 5: Input Feature Variability Analysis (UES) -------------------------
# Plot UES for each feature dimension.
try:
    fig, axs = plt.subplots(1, n4, figsize=(4 * n4, 4), dpi=dpi)
    if n4 == 1:
        axs = [axs]
    for i, key in enumerate(keys4):
        ues_curve = feat_var[key]["metrics"]["train"]
        axs[i].plot(ues_curve, label="UES", color="orange")
        axs[i].set_title(f"{key} UES")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("UES")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig5_feature_variability_ues.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 5 (Feature Variability UES):", e)


# ------------------------- Figure 6: Dataset Diversity Analysis (Loss) -------------------------
# From the Dataset Diversity Analysis experiment, plot training loss for each dataset type.
try:
    file6 = "experiment_results/experiment_1af7f8de5d3743e785510e5245e7b256_proc_2540414/experiment_data.npy"
    data6 = np.load(file6, allow_pickle=True).item()
    diversity = data6["Dataset_Diversity_Analysis"]
    keys6 = sorted(diversity.keys())  # e.g., type1, type2, type3
    n6 = len(keys6)
    fig, axs = plt.subplots(1, n6, figsize=(4 * n6, 4), dpi=dpi)
    if n6 == 1:
        axs = [axs]
    for i, key in enumerate(keys6):
        loss_curve = diversity[key]["losses"]["train"]
        axs[i].plot(loss_curve, label="Loss")
        axs[i].set_title(f"{key} Training Loss")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig6_dataset_diversity_loss.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 6 (Dataset Diversity Loss):", e)


# ------------------------- Figure 7: Dataset Diversity Analysis (UES) -------------------------
# Plot UES for each dataset type in the Dataset Diversity Analysis.
try:
    fig, axs = plt.subplots(1, n6, figsize=(4 * n6, 4), dpi=dpi)
    if n6 == 1:
        axs = [axs]
    for i, key in enumerate(keys6):
        ues_curve = diversity[key]["metrics"]["train"]
        axs[i].plot(ues_curve, label="UES", color="orange")
        axs[i].set_title(f"{key} UES")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("UES")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig7_dataset_diversity_ues.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 7 (Dataset Diversity UES):", e)


# ------------------------- Figure 8: Input Noise Robustness Analysis (Loss Comparison) -------------------------
# Compare training loss on clean vs noisy datasets from the noise robustness experiment.
try:
    file8 = "experiment_results/experiment_ef5c2741bbf844f39e992773955a9d5c_proc_2540412/experiment_data.npy"
    data8 = np.load(file8, allow_pickle=True).item()
    noise_data = data8["input_noise_robustness"]
    loss_clean = noise_data["clean_dataset"]["losses"]["train"]
    loss_noisy = noise_data["noisy_dataset"]["losses"]["train"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    axs[0].plot(loss_clean, label="Clean Loss")
    axs[0].set_title("Clean Dataset Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    set_spines(axs[0])
    axs[1].plot(loss_noisy, label="Noisy Loss", color="red")
    axs[1].set_title("Noisy Dataset Training Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    set_spines(axs[1])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig8_noise_robustness_loss.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 8 (Noise Robustness Loss):", e)


# ------------------------- Figure 9: Input Noise Robustness Analysis (UES Comparison) -------------------------
# Compare UES metrics on clean vs noisy datasets.
try:
    ues_clean = noise_data["clean_dataset"]["metrics"]["train"]
    ues_noisy = noise_data["noisy_dataset"]["metrics"]["train"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    axs[0].plot(ues_clean, label="Clean UES")
    axs[0].set_title("Clean Dataset UES")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("UES")
    axs[0].legend()
    set_spines(axs[0])
    axs[1].plot(ues_noisy, label="Noisy UES", color="red")
    axs[1].set_title("Noisy Dataset UES")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("UES")
    axs[1].legend()
    set_spines(axs[1])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig9_noise_robustness_ues.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 9 (Noise Robustness UES):", e)


# ------------------------- Figure 10: Multi-Dataset Evaluation with Diverse Task Types (Loss) -------------------------
# For tasks such as Sentiment Analysis, Question Answering, Text Summarization.
try:
    file10 = "experiment_results/experiment_47e273272b6c486eb580b669158d0cd8_proc_2540413/experiment_data.npy"
    data10 = np.load(file10, allow_pickle=True).item()
    multi_task = data10["multi_dataset_study"]
    tasks = ["sentiment_analysis", "question_answering", "text_summarization"]
    n10 = len(tasks)
    fig, axs = plt.subplots(1, n10, figsize=(4 * n10, 4), dpi=dpi)
    if n10 == 1:
        axs = [axs]
    for i, t in enumerate(tasks):
        loss_task = multi_task[t]["losses"]["train"]
        # Replace underscores with spaces and title-case the task names
        axs[i].plot(loss_task, label="Loss")
        axs[i].set_title(f"{t.replace('_', ' ').title()} Loss")
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig10_multi_task_loss.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 10 (Multi-Task Loss):", e)


# ------------------------- Figure 11: Multi-Dataset Performance Consistency Evaluation (Scatter) -------------------------
# Scatter plots comparing predictions vs. ground truth for each dataset type.
try:
    file11 = "experiment_results/experiment_6d4fd3f87dd44fd0822c7b389def9296_proc_2540414/experiment_data.npy"
    data11 = np.load(file11, allow_pickle=True).item()
    consistency = data11["multi_dataset_evaluation"]
    keys11 = sorted(consistency.keys())
    n11 = len(keys11)
    fig, axs = plt.subplots(1, n11, figsize=(5 * n11, 4), dpi=dpi)
    if n11 == 1:
        axs = [axs]
    for i, key in enumerate(keys11):
        preds = np.array(consistency[key]["predictions"])
        gt = np.array(consistency[key]["ground_truth"])
        axs[i].scatter(gt[:, 0], gt[:, 1], label="Ground Truth", alpha=0.5)
        axs[i].scatter(preds[:, 0], preds[:, 1], label="Predictions", alpha=0.5, color="red")
        axs[i].set_title(f"{key} Predictions vs Truth")
        axs[i].set_xlabel("Value 1")
        axs[i].set_ylabel("Value 2")
        axs[i].legend()
        set_spines(axs[i])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig11_consistency_scatter.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 11 (Consistency Scatter):", e)


# ------------------------- Figure 12: Input Feature Dimensionality Analysis -------------------------
# Plot training loss and UES metric (side-by-side) for the synthetic dataset.
try:
    file12 = "experiment_results/experiment_e2a018e94eff41cf871a7a669e8ad687_proc_2540412/experiment_data.npy"
    data12 = np.load(file12, allow_pickle=True).item()
    dim_analysis = data12["input_feature_dimensionality_analysis"]["synthetic_dataset"]
    loss_dim = dim_analysis["losses"]["train"]
    ues_dim = dim_analysis["metrics"]["train"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    axs[0].plot(loss_dim, label="Loss")
    axs[0].set_title("Dimensionality Analysis: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    set_spines(axs[0])
    axs[1].plot(ues_dim, label="UES", color="orange")
    axs[1].set_title("Dimensionality Analysis: UES")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("UES")
    axs[1].legend()
    set_spines(axs[1])
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "fig12_dimensionality_analysis.png"))
    plt.close(fig)
except Exception as e:
    print("Error in Figure 12 (Dimensionality Analysis):", e)


print("All final figures have been generated and saved in the 'figures' directory.")