import numpy as np
import torch
import random
from datasets import load_dataset

# Set global RNG seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print("Global seeds set to", seed)

##############################################
# 1. Synthetic Binomial Loss Data Preparation
##############################################
# Parameters for binomial loss experiment:
# n: number of samples per trial, K: number of draws per sample,
# candidate lambdas over [0,1] and M: trials
n = 10      
K = 4       
M = 10000  
candidate_lambdas = np.linspace(0.0, 1.0, K + 1)

# Generate uniform draws: shape (M, n, K)
uniform_values = np.random.uniform(0, 1, size=(M, n, K))
# Compute losses along each candidate lambda:
# loss = average indicator {V > lambda} over the K draws for each sample
loss_matrix = np.empty((M, n, len(candidate_lambdas)))
for j, lam in enumerate(candidate_lambdas):
    loss_matrix[:, :, j] = (uniform_values > lam).mean(axis=-1)
# Check monotonicity: losses should be non-increasing in lambda for each sample
assert np.all(np.diff(loss_matrix, axis=-1) <= 1e-6), "Loss function not monotonic!"
print("Synthetic Binomial Loss Data prepared with shape:", loss_matrix.shape)
print("Candidate lambdas:", candidate_lambdas)

#####################################################
# 2. Synthetic Heteroskedastic Regression Data Setup
#####################################################
# Calibration data parameters: n_calib points, thresholds for loss computed as 1{|Y| > lambda}
n_calib = 200  
N_test = 200000  
lambda_grid_reg = np.linspace(0, 5, 11) 

# Calibration dataset generation:
X_calib = np.random.uniform(0, 4, size=(n_calib, 1))
# Variance equals the value of X, so std deviation is X (simulate heteroskedastic noise)
Y_calib = np.random.normal(0, X_calib[:, 0])

# Compute calibration losses for each candidate lambda: loss = indicator(|Y| > lambda)
calib_losses = np.array([ (np.abs(Y_calib) > lam).astype(float) for lam in lambda_grid_reg ])
# Verify monotonicity across lambda for each sample:
assert np.all(np.diff(calib_losses, axis=0) <= 1e-6), "Regression losses not monotonic!"
print("Heteroskedastic Regression Calibration Data: X shape =", X_calib.shape, ", Y shape =", Y_calib.shape)
print("Regression candidate thresholds:", lambda_grid_reg)

# Generate large test set for risk estimation:
X_test = np.random.uniform(0, 4, size=(N_test, 1))
Y_test = np.random.normal(0, X_test[:, 0])
# Estimate risk: fraction of examples where |Y| > lambda for each candidate threshold
risk_estimates = np.array([ np.mean(np.abs(Y_test) > lam) for lam in lambda_grid_reg ])
print("Estimated ground-truth risk per candidate threshold:", risk_estimates)

#####################################################
# 3. MS-COCO Multilabel Classification Dataset Setup
#####################################################
# Instead of using the problematic "coco" dataset, we load "coco_captions" with 2017 configuration.
# This dataset has image information and captions; we'll simulate multilabel classification scores.
try:
    coco_data = load_dataset("coco_captions", "2017", split="validation")
    print("Loaded coco_captions (2017) validation split with", len(coco_data), "examples.")
except Exception as e:
    print("Error loading coco_captions dataset:", e)
    coco_data = None

if coco_data is not None:
    # Filter examples with a caption (simulate requiring the existence of annotations)
    coco_data = coco_data.filter(lambda ex: ex.get("caption") is not None)
    # Add fake multilabel scores: simulate 80 classes with sigmoid scores between 0 and 1
    def add_fake_multilabel(example):
        example["scores"] = np.random.uniform(0, 1, size=80).tolist()
        return example
    coco_data = coco_data.map(add_fake_multilabel)
    # Precompute sorted scores (descending) for possible threshold sweeps.
    def compute_sorted_scores(example):
        example["sorted_scores"] = sorted(example["scores"], reverse=True)
        return example
    coco_data = coco_data.map(compute_sorted_scores)
    print("After processing, coco dataset sample keys:", coco_data.column_names)
else:
    print("Skipping MS-COCO multilabel processing due to dataset load failure.")

##############################################
# Summary of Prepared Data
##############################################
print("\n--- Data Preparation Summary ---")
print("Synthetic Binomial Loss: Trials =", loss_matrix.shape[0], ", Samples per trial =", loss_matrix.shape[1])
print("Heteroskedastic Regression: Calibration X shape =", X_calib.shape, ", Test set X shape =", X_test.shape)
if coco_data is not None:
    print("MS-COCO Multilabel: Number of examples =", len(coco_data))
else:
    print("MS-COCO Multilabel: Dataset not available.")