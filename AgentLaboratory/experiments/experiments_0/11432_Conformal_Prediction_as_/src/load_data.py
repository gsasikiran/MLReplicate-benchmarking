import numpy as np
import random
import torch
import os
from datasets import load_dataset
from scipy.stats import dirichlet

# Set reproducible seeds for Python, numpy, and torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
print("Global seeds have been set for reproducibility.")

###############################
# Part 1: Synthetic Binomial Loss
###############################
n_binom = 10       # number of calibration samples
K_binom = 4        # repetitions per sample
alpha_binom = 0.4  # threshold for evaluation
beta_binom = 0.95  # failure probability level
B = 1              # loss bound (maximum loss)
M_trials = 10000   # number of Monte Carlo trials
lambda_grid = np.linspace(0, 1, 101)  # candidate lambda values

# Generate calibration data once so monotonicity holds
V = np.random.uniform(0, 1, (n_binom, K_binom))
# Build loss matrix: for each lambda value and each calibration example,
# compute ℓ_i(λ)=mean(1{V_ik > λ}) which is monotone non-increasing in λ.
loss_matrix_binom = []
for lam in lambda_grid:
    losses = (V > lam).mean(axis=1)
    loss_matrix_binom.append(losses)
loss_matrix_binom = np.array(loss_matrix_binom)  # shape: (num_lambda, n_binom)

# Check monotonicity for each calibration example (loss should not increase with lambda)
mono_flag = True
for j in range(n_binom):
    if not np.all(np.diff(loss_matrix_binom[:, j]) <= 1e-8):
        mono_flag = False
        break
print("Synthetic Binomial Loss: Monotonicity check (should be True):", mono_flag)

# For demonstration, pick a sample lambda value (choose middle of grid)
lam_ex_binom = lambda_grid[len(lambda_grid)//2]
# Use the corresponding losses from the fixed V matrix
losses_binom = (V > lam_ex_binom).mean(axis=1)
# Sort losses and append B as the (n+1)-th element
sorted_losses_binom = np.sort(losses_binom)
sorted_losses_binom = np.concatenate([sorted_losses_binom, [B]])
# Draw Dirichlet samples (dimension = n_binom+1)
dirichlet_samples_binom = np.random.dirichlet(np.ones(n_binom + 1), size=M_trials)
# Compute weighted sum L+ for each trial
L_plus_binom = (dirichlet_samples_binom * sorted_losses_binom).sum(axis=1)
# Estimate probability that L+ <= alpha_binom
prob_binom = np.mean(L_plus_binom <= alpha_binom)
print("Synthetic Binomial Loss Data Preparation:")
print("For lambda =", lam_ex_binom, "estimated Pr(L+ <= alpha):", prob_binom)

###############################################
# Part 2: Synthetic Heteroskedastic Regression
###############################################
n_reg = 200       # calibration sample size
alpha_reg = 0.1   # threshold for miscoverage
beta_reg = 0.95   # failure probability level
N_test = 200000   # size of the independent test set

# Generate calibration data: X ~ Uniform[0,4] and Y|X ~ Normal(0, X^2)
X_calib = np.random.uniform(0, 4, n_reg)
Y_calib = np.array([np.random.normal(0, x**2) for x in X_calib])
# For each candidate lambda, compute loss ℓ_i(λ)=1{|Y_calib| > λ}
loss_dict_reg = {}
loss_matrix_reg = []
for lam in lambda_grid:
    losses = (np.abs(Y_calib) > lam).astype(float)
    loss_dict_reg[lam] = losses  # store in dictionary for later use
    loss_matrix_reg.append(losses)
loss_matrix_reg = np.array(loss_matrix_reg)  # shape: (num_lambda, n_reg)

# Check monotonicity across lambda for the first calibration point
if np.all(np.diff(loss_matrix_reg[:, 0]) <= 1e-8):
    mono_reg = True
else:
    mono_reg = False
print("Synthetic Heteroskedastic Regression Loss: Monotonicity check for first sample (should be True):", mono_reg)

# Select a demonstration lambda value (middle of grid)
lam_ex_reg = lambda_grid[len(lambda_grid)//2]
losses_reg_ex = loss_dict_reg[lam_ex_reg]
sorted_losses_reg = np.sort(losses_reg_ex)
sorted_losses_reg = np.concatenate([sorted_losses_reg, [B]])
# Monte Carlo Dirichlet sampling for regression
dirichlet_samples_reg = np.random.dirichlet(np.ones(n_reg + 1), size=M_trials)
L_plus_reg = (dirichlet_samples_reg * sorted_losses_reg).sum(axis=1)
prob_reg = np.mean(L_plus_reg <= alpha_reg)
print("\nSynthetic Heteroskedastic Regression Data Preparation:")
print("For lambda =", lam_ex_reg, "estimated Pr(L+ <= alpha):", prob_reg)

# Ground-truth risk estimation using a large independent test set
X_test = np.random.uniform(0, 4, N_test)
Y_test = np.array([np.random.normal(0, x**2) for x in X_test])
# To get the risk at the test lambda, use the lambda closest to our chosen lam_ex_reg
risk_est = np.mean(np.abs(Y_test) > lam_ex_reg)
print("Estimated ground-truth risk R̂(lambda) for lambda =", lam_ex_reg, "is:", risk_est)

##################################################
# Part 3: MS-COCO Multilabel FNR Data Preparation
##################################################
# Try to load the MS-COCO dataset from HuggingFace.
# Since the "coco" dataset with configuration "2014" may be inaccessible, we attempt with configuration "2017".
try:
    # Attempt to load MS-COCO 2017 validation split
    coco_data = load_dataset("coco", "2017", split="validation")
    # In this dataset, assume images with annotations have a non-empty 'annotations' field.
    def has_annotations(example):
        return ("annotations" in example) and (example["annotations"] is not None) and (len(example["annotations"]) > 0)
    coco_data = coco_data.filter(has_annotations)
    print("\nMS-COCO Data Preparation:")
    print("Loaded MS-COCO 2017 validation split with", len(coco_data), "examples after filtering.")
    
    # Simulate pre-computed sigmoid scores for multilabel classification.
    # Assume 80 classes; add a dummy field 'sigmoid_scores' with random values in [0, 1].
    num_classes = 80
    coco_data = coco_data.map(lambda x: {"sigmoid_scores": np.random.uniform(0, 1, num_classes).tolist()})
    print("MS-COCO data prepared with dummy sigmoid scores for multilabel FNR experiment.")
except Exception as e:
    print("Error loading MS-COCO dataset:", e)

print("\nData preparation complete for all experiments.")