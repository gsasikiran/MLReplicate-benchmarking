#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# Set a larger font size and other stylistic parameters for publication quality figures.
plt.rcParams.update({'font.size': 14})

# Ensure the final figures are saved in the "figures" directory.
os.makedirs("figures", exist_ok=True)

# Utility to remove top and right spines from an axis.
def style_ax(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)

# ------------------------------
# Figure 1: Baseline Experiment (Dropout Tuning)
# Data file: experiment_results/experiment_4d96c2e22105498291d2a6f348af6da5_proc_2536319/experiment_data.npy
try:
    baseline_path = "experiment_results/experiment_4d96c2e22105498291d2a6f348af6da5_proc_2536319/experiment_data.npy"
    baseline_data = np.load(baseline_path, allow_pickle=True).item()
    
    losses = baseline_data["dropout_tuning"]["synthetic_dataset"]["losses"]["train"]
    cods = baseline_data["dropout_tuning"]["synthetic_dataset"]["metrics"]["train"]
    epochs = np.arange(1, len(losses) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    
    # Training Loss
    axs[0].plot(epochs, losses, label="Training Loss", color="blue")
    axs[0].set_title("Baseline: Training Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    style_ax(axs[0])
    
    # CODS Metric
    axs[1].plot(epochs, cods, label="Training CODS", color="orange")
    axs[1].set_title("Baseline: CODS over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("CODS")
    axs[1].legend()
    style_ax(axs[1])
    
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "baseline_dropout_tuning.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Baseline plot: {e}")

# ------------------------------
# Figure 2: Ablation - Sequence Length Variation
# Data file: experiment_results/experiment_4371fb9468684dda9989a48cc7cf7564_proc_2537107/experiment_data.npy
try:
    seq_path = "experiment_results/experiment_4371fb9468684dda9989a48cc7cf7564_proc_2537107/experiment_data.npy"
    seq_data = np.load(seq_path, allow_pickle=True).item()
    
    seq_variants = seq_data["sequence_length_variation"]
    seq_keys = list(seq_variants.keys())
    
    # Prepare a color map for consistency
    colors = ['blue', 'green', 'red']
    
    # Aggregate training losses and CODS metrics from all sequence lengths.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    
    for i, seq in enumerate(seq_keys):
        train_loss = seq_variants[seq]["losses"]["train"]
        cods_metric = seq_variants[seq]["metrics"]["train"]
        epochs_loss = np.arange(1, len(train_loss) + 1)
        epochs_cods = np.arange(1, len(cods_metric) + 1)
        
        axs[0].plot(epochs_loss, train_loss, label=f"Seq Length {seq}", color=colors[i % len(colors)])
        axs[1].plot(epochs_cods, cods_metric, label=f"Seq Length {seq}", color=colors[i % len(colors)])
    
    axs[0].set_title("Seq Length Variation: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    style_ax(axs[0])
    
    axs[1].set_title("Seq Length Variation: CODS Metric")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("CODS")
    axs[1].legend()
    style_ax(axs[1])
    
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_sequence_length_variation.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Sequence Length Variation plot: {e}")

# ------------------------------
# Figure 3: Ablation - Embedding Dimension Variation
# Data file: experiment_results/experiment_835a85e62a8144cf84e5c8480def4b6f_proc_2537108/experiment_data.npy
try:
    emb_path = "experiment_results/experiment_835a85e62a8144cf84e5c8480def4b6f_proc_2537108/experiment_data.npy"
    emb_data = np.load(emb_path, allow_pickle=True).item()
    
    emb_group = emb_data["embedding_variation"]["synthetic_dataset"]
    emb_losses = emb_group["losses"]["train"]
    emb_cods = emb_group["metrics"]["train"]
    epochs_emb = np.arange(1, len(emb_losses) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12,4), dpi=300)
    
    axs[0].plot(epochs_emb, emb_losses, label="Training Loss", color="blue")
    axs[0].set_title("Embedding Variation: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    style_ax(axs[0])
    
    axs[1].plot(epochs_emb, emb_cods, label="CODS Metric", color="orange")
    axs[1].set_title("Embedding Variation: CODS over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("CODS")
    axs[1].legend()
    style_ax(axs[1])
    
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_embedding_variation.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Embedding Dimension Variation plot: {e}")

# ------------------------------
# Figure 4: Ablation - Hidden Layer Size Variation
# Data file: experiment_results/experiment_7fee375583034e97b0166aa81bb04eef_proc_2537107/experiment_data.npy
try:
    hidden_path = "experiment_results/experiment_7fee375583034e97b0166aa81bb04eef_proc_2537107/experiment_data.npy"
    hidden_data = np.load(hidden_path, allow_pickle=True).item()
    
    hidden_group = hidden_data["hidden_layer_size_variation"]["synthetic_dataset"]
    hidden_losses = hidden_group["losses"]
    hidden_cods = hidden_group["metrics"]
    epochs_hidden = np.arange(1, len(hidden_losses) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12,4), dpi=300)
    
    axs[0].plot(epochs_hidden, hidden_losses, label="Training Loss", color="blue")
    axs[0].set_title("Hidden Layer Size Variation: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    style_ax(axs[0])
    
    axs[1].plot(epochs_hidden, hidden_cods, label="CODS Metric", color="orange")
    axs[1].set_title("Hidden Layer Size Variation: CODS over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("CODS")
    axs[1].legend()
    style_ax(axs[1])
    
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_hidden_layer_variation.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Hidden Layer Size Variation plot: {e}")

# ------------------------------
# Figure 5: Ablation - Activation Function Variation
# Data file: experiment_results/experiment_b5905eeb42624900b8992e33d3e61630_proc_2537105/experiment_data.npy
try:
    act_path = "experiment_results/experiment_b5905eeb42624900b8992e33d3e61630_proc_2537105/experiment_data.npy"
    act_data = np.load(act_path, allow_pickle=True).item()
    
    act_group = act_data["activation_variation"]["synthetic_dataset"]
    act_losses = act_group["losses"]["train"]
    act_cods = act_group["metrics"]["train"]
    epochs_act = np.arange(1, len(act_losses) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12,4), dpi=300)
    
    axs[0].plot(epochs_act, act_losses, label="Training Loss", color="blue")
    axs[0].set_title("Activation Variation: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    style_ax(axs[0])
    
    axs[1].plot(epochs_act, act_cods, label="CODS Metric", color="orange")
    axs[1].set_title("Activation Variation: CODS over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("CODS")
    axs[1].legend()
    style_ax(axs[1])
    
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_activation_variation.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Activation Function Variation plot: {e}")

# ------------------------------
# Figure 6: Ablation - Multiple Training Epochs Variation
# Data file: experiment_results/experiment_b338b19349bc4818928896006eec2167_proc_2537107/experiment_data.npy
try:
    epochs_path = "experiment_results/experiment_b338b19349bc4818928896006eec2167_proc_2537107/experiment_data.npy"
    epochs_data = np.load(epochs_path, allow_pickle=True).item()
    
    mult_group = epochs_data["multiple_epochs_variation"]["synthetic_dataset"]
    mult_losses = mult_group["losses"]["train"]
    mult_cods = mult_group["metrics"]["train"]
    epochs_range = np.arange(1, len(mult_losses) + 1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12,4), dpi=300)
    
    axs[0].plot(epochs_range, mult_losses, label="Training Loss", color="blue")
    axs[0].set_title("Multiple Epochs Variation: Training Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    style_ax(axs[0])
    
    axs[1].plot(epochs_range, mult_cods, label="CODS Metric", color="orange")
    axs[1].set_title("Multiple Epochs Variation: CODS over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("CODS")
    axs[1].legend()
    style_ax(axs[1])
    
    plt.tight_layout()
    fig.savefig(os.path.join("figures", "ablation_multiple_epochs_variation.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error in Multiple Training Epochs Variation plot: {e}")

print("All plots generated and saved in the 'figures' directory.")