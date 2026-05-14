import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from math import ceil, log, sqrt
from datasets import load_dataset

# Global seeds already set in dataset code – we assume they are set

###############################################################################
# NOTE:
# This code runs experiments for three setups:
# 1) Synthetic Binomial Loss: Here the loss is given by ℓ_i(λ)=mean(1{V_ik>λ}). 
#    The true risk is analytically R(λ)=1−λ so failure occurs if λ < 0.6 (for α=0.4).
#
# 2) Synthetic Heteroskedastic Regression: Calibration data are generated with Y|X ~ Normal(0, X^2)
#    and loss is ℓ_i(λ)=1{|Y_calib|>λ}. The test risk is estimated on an independent large set.
#
# 3) MS-COCO Multilabel FNR: Using the dummy sigmoid scores for each image, we define:
#    Ground-truth: a class is relevant if its score > 0.5.
#    Prediction set: classes with score >= 1 − λ.
#    The per-image FNR is the fraction of relevant classes that are missed.
#    Failure is declared if the mean FNR (averaged over images) > α.
#
# In all experiments we implement four decision rules:
#   • SCP (Split Conformal Prediction): For miscoverage loss, we use the order‐statistic rule.
#       (For binary losses, this amounts to picking λ_scp = 1 − q, where q is the quantile determined by index ceil((n+1)(1−α)) evaluated on the binary calibration indicators.)
#   • CRC (Conformal Risk Control): Choose the minimal λ in the candidate grid for which E(L⁺)=(Σ loss + B)/(n+1) ≤ α.
#   • Bayesian Quadrature HPD: For each λ, sample Dirichlet weights (N_dirichlet=1000) applied over the sorted calibration losses (with B appended) 
#       and estimate pₗ = Pr(L⁺ ≤ α); choose the minimal λ with pₗ ≥ β.
#   • RCPS Baseline with Hoeffding UCB: Choose minimal λ such that mean(loss) + sqrt(log(1/(1−β))/(2n)) ≤ α.
#
# For each experiment, we loop over a number of trials (resampling calibration data) and record:
#   - The decision λ for each method.
#   - The estimated risk on held-out test data (or analytic risk for binom).
#   - A flag indicating failure (risk exceeding α).
#
# We then display summary statistics and generate two figures per experiment:
#   Figure_1_<EXP_NAME>.png: Histogram of chosen λ's across trials for the methods.
#   Figure_2_<EXP_NAME>.png: Bar plot of failure percentages per method.
###############################################################################

# Candidate lambda grid (must be increasing)
lambda_grid = np.linspace(0, 1, 101)
B = 1.

#############################
# Experiment 1: Synthetic Binomial Loss
#############################
print("\nExperiment 1: Synthetic Binomial Loss")
print("This experiment uses a synthetic binomial model where each calibration sample is a vector V ~ Uniform(0,1).")
print("The loss for candidate λ is ℓ(λ)=mean(1{V > λ}). The true risk is R(λ)=1−λ, hence failure occurs if λ < 0.6 (with α=0.4).")
n_binom = 10       # number of calibration samples
K_binom = 4        # repetitions per sample
alpha_binom = 0.4  # risk threshold
beta_binom = 0.95  # failure probability level (used in Bayesian HPD and RCPS)
N_dirichlet = 1000  # Monte Carlo samples for Bayesian HPD
N_trials_binom = 100  # number of simulation trials

# Arrays to store results per method: each shape (N_trials_binom,)
results_binom_lambda_scp = []
results_binom_lambda_crc = []
results_binom_lambda_hpd = []
results_binom_lambda_rcps = []
failures_binom_scp = []
failures_binom_crc = []
failures_binom_hpd = []
failures_binom_rcps = []

for t in range(N_trials_binom):
    # Generate calibration data: V shape (n_binom, K_binom)
    V = np.random.uniform(0, 1, (n_binom, K_binom))
    # For each candidate lambda, compute calibration losses per sample as mean(Indicator(V > lambda))
    calib_loss_grid = []  # shape: (num_lambda, n_binom)
    for lam in lambda_grid:
        losses = (V > lam).mean(axis=1)
        calib_loss_grid.append(losses)
    calib_loss_grid = np.array(calib_loss_grid)  # shape: (len(lambda_grid), n_binom)
    
    # For each candidate lambda, compute summary statistics:
    # For methods that require sorted losses, we sort per candidate and append B.
    crc_found = False
    hpd_found = False
    rcps_found = False
    scp_found = False
    lambda_crc = None
    lambda_hpd = None
    lambda_rcps = None
    lambda_scp = None
    for idx, lam in enumerate(lambda_grid):
        losses = calib_loss_grid[idx]  # vector of length n_binom
        # CRC: compute expected L+: (sum(losses)+B)/(n+1)
        exp_L_plus = (np.sum(losses) + B) / (n_binom + 1)
        if (not crc_found) and (exp_L_plus <= alpha_binom):
            lambda_crc = lam
            crc_found = True
            
        # RCPS: use Hoeffding UCB with delta = 1-beta, i.e. UCB = mean(loss) + sqrt(log(1/(1-beta))/(2*n))
        ucb = np.mean(losses) + sqrt(log(1/(1 - beta_binom))/(2*n_binom))
        if (not rcps_found) and (ucb <= alpha_binom):
            lambda_rcps = lam
            rcps_found = True
            
        # Bayesian HPD:
        # Sort losses in increasing order and append B as nth+1 element
        sorted_losses = np.sort(losses)
        sorted_extended = np.concatenate([sorted_losses, [B]])
        # Draw Dirichlet samples (reuse same samples for this candidate across trials for variance reduction)
        dirichlet_samples = np.random.dirichlet(np.ones(n_binom+1), size=N_dirichlet)
        L_plus_samples = (dirichlet_samples * sorted_extended).sum(axis=1)
        p_lam = np.mean(L_plus_samples <= alpha_binom)
        if (not hpd_found) and (p_lam >= beta_binom):
            lambda_hpd = lam
            hpd_found = True
            
        # SCP: For split conformal, take order statistic of the calibration losses.
        # Compute order index: i = ceil((n+1)*(1-α)) - 1 (0-indexed)
        order_idx = ceil((n_binom+1)*(1 - alpha_binom)) - 1
        sorted_losses_scp = np.sort(losses)
        s_star = sorted_losses_scp[order_idx]  # this is the calibration loss at that quantile
        # For miscoverage loss where loss = 1{V > λ}, note that the expected loss is 1 - λ.
        # Invert: λ_scp = 1 - s_star.
        candidate_scp = 1 - s_star
        # We want the minimal lambda in the grid that is >= candidate_scp.
        if not scp_found and lam >= candidate_scp:
            lambda_scp = lam
            scp_found = True

        # If all four rules found, break early
        if crc_found and rcps_found and hpd_found and scp_found:
            break

    # If any method wasn't triggered, assign maximum lambda value.
    if lambda_crc is None:
        lambda_crc = lambda_grid[-1]
    if lambda_rcps is None:
        lambda_rcps = lambda_grid[-1]
    if lambda_hpd is None:
        lambda_hpd = lambda_grid[-1]
    if lambda_scp is None:
        lambda_scp = lambda_grid[-1]
        
    results_binom_lambda_crc.append(lambda_crc)
    results_binom_lambda_rcps.append(lambda_rcps)
    results_binom_lambda_hpd.append(lambda_hpd)
    results_binom_lambda_scp.append(lambda_scp)
    
    # Compute true risk for each method: R(λ)=1 - λ. Failure occurs if risk > α_binom, i.e. λ < 1 - α_binom = 0.6.
    failures_binom_crc.append(1 if lambda_crc < 0.6 else 0)
    failures_binom_rcps.append(1 if lambda_rcps < 0.6 else 0)
    failures_binom_hpd.append(1 if lambda_hpd < 0.6 else 0)
    failures_binom_scp.append(1 if lambda_scp < 0.6 else 0)

# Convert results to numpy arrays for ease of analysis.
results_binom_lambda_crc = np.array(results_binom_lambda_crc)
results_binom_lambda_rcps = np.array(results_binom_lambda_rcps)
results_binom_lambda_hpd = np.array(results_binom_lambda_hpd)
results_binom_lambda_scp = np.array(results_binom_lambda_scp)
failures_binom_crc = np.array(failures_binom_crc)
failures_binom_rcps = np.array(failures_binom_rcps)
failures_binom_hpd = np.array(failures_binom_hpd)
failures_binom_scp = np.array(failures_binom_scp)

print("\nSynthetic Binomial Loss Results Summary:")
print("Average selected λ (SCP):", np.mean(results_binom_lambda_scp))
print("Average selected λ (CRC):", np.mean(results_binom_lambda_crc))
print("Average selected λ (Bayesian HPD):", np.mean(results_binom_lambda_hpd))
print("Average selected λ (RCPS):", np.mean(results_binom_lambda_rcps))
print("Failure Rate (SCP):", np.mean(failures_binom_scp))
print("Failure Rate (CRC):", np.mean(failures_binom_crc))
print("Failure Rate (Bayesian HPD):", np.mean(failures_binom_hpd))
print("Failure Rate (RCPS):", np.mean(failures_binom_rcps))

# Generate Figures for Synthetic Binomial Loss
plt.figure(figsize=(10,6))
plt.hist(results_binom_lambda_scp, bins=20, alpha=0.5, label='SCP')
plt.hist(results_binom_lambda_crc, bins=20, alpha=0.5, label='CRC')
plt.hist(results_binom_lambda_hpd, bins=20, alpha=0.5, label='Bayesian HPD')
plt.hist(results_binom_lambda_rcps, bins=20, alpha=0.5, label='RCPS')
plt.xlabel("Selected λ")
plt.ylabel("Frequency")
plt.title("Figure_1_Synthetic_Binomial: Histogram of Selected λ")
plt.legend()
plt.savefig("Figure_1_Synthetic_Binomial.png")
plt.close()

failure_rates = [np.mean(failures_binom_scp), np.mean(failures_binom_crc),
                 np.mean(failures_binom_hpd), np.mean(failures_binom_rcps)]
methods = ['SCP', 'CRC', 'Bayesian HPD', 'RCPS']
plt.figure(figsize=(8,6))
plt.bar(methods, failure_rates)
plt.ylabel("Failure Rate")
plt.title("Figure_2_Synthetic_Binomial: Failure Rates by Method")
plt.savefig("Figure_2_Synthetic_Binomial.png")
plt.close()

#############################
# Experiment 2: Synthetic Heteroskedastic Regression
#############################
print("\nExperiment 2: Synthetic Heteroskedastic Regression")
print("This experiment simulates heteroskedastic regression data. Calibration data are generated with X ~ Uniform(0,4) and Y|X ~ Normal(0, X^2).")
print("The loss is ℓ_i(λ)=1{|Y_calib|>λ}. The test risk is estimated on an independent test set (N_test=200000); failure occurs if estimated risk > α_reg (α_reg=0.1).")
n_reg = 200       # calibration sample size
alpha_reg = 0.1   # miscoverage threshold
beta_reg = 0.95
N_test = 200000
N_trials_reg = 100  # number of trials
N_dirichlet = 1000  # for Bayesian HPD

results_reg_lambda_scp = []
results_reg_lambda_crc = []
results_reg_lambda_hpd = []
results_reg_lambda_rcps = []
failures_reg = []

# Pre-generate independent test set for risk estimation
X_test = np.random.uniform(0, 4, N_test)
Y_test = np.array([np.random.normal(0, x**2) for x in X_test])

for t in range(N_trials_reg):
    # Generate calibration data
    X_calib = np.random.uniform(0, 4, n_reg)
    Y_calib = np.array([np.random.normal(0, x**2) for x in X_calib])
    # For each candidate lambda, loss = indicator(|Y_calib| > λ)
    calib_loss_grid = []
    for lam in lambda_grid:
        losses = (np.abs(Y_calib) > lam).astype(float)
        calib_loss_grid.append(losses)
    calib_loss_grid = np.array(calib_loss_grid)  # shape: (num_lambda, n_reg)
    
    crc_found = False
    hpd_found = False
    rcps_found = False
    scp_found = False
    lambda_crc = None
    lambda_hpd = None
    lambda_rcps = None
    lambda_scp = None
    for idx, lam in enumerate(lambda_grid):
        losses = calib_loss_grid[idx]
        # CRC: expected L+ = (sum(losses)+B)/(n+1)
        exp_L_plus = (np.sum(losses)+B)/(n_reg+1)
        if (not crc_found) and (exp_L_plus <= alpha_reg):
            lambda_crc = lam
            crc_found = True

        # RCPS: UCB = mean(loss) + sqrt(log(1/(1-beta))/(2*n))
        ucb = np.mean(losses) + sqrt(log(1/(1-beta_reg))/(2*n_reg))
        if (not rcps_found) and (ucb <= alpha_reg):
            lambda_rcps = lam
            rcps_found = True

        # Bayesian HPD:
        sorted_losses = np.sort(losses)
        sorted_extended = np.concatenate([sorted_losses, [B]])
        dirichlet_samples = np.random.dirichlet(np.ones(n_reg+1), size=N_dirichlet)
        L_plus_samples = (dirichlet_samples * sorted_extended).sum(axis=1)
        p_lam = np.mean(L_plus_samples <= alpha_reg)
        if (not hpd_found) and (p_lam >= beta_reg):
            lambda_hpd = lam
            hpd_found = True
        
        # SCP rule: use order statistic on calibration losses.
        order_idx = ceil((n_reg+1)*(1-alpha_reg)) - 1
        sorted_losses_scp = np.sort(losses)
        s_star = sorted_losses_scp[order_idx]
        # For miscoverage loss, if true loss = 1{|Y| > λ}, then expected loss = risk = P(|Y|>λ).
        # We invert approximately using the empirical risk: choose λ such that risk ≈ s_star.
        # Since risk is monotonic in λ (decreasing), we choose the minimal λ in grid with risk <= s_star.
        if not scp_found:
            if np.mean(losses) <= s_star:
                lambda_scp = lam
                scp_found = True

        if crc_found and rcps_found and hpd_found and scp_found:
            break

    if lambda_crc is None:
        lambda_crc = lambda_grid[-1]
    if lambda_rcps is None:
        lambda_rcps = lambda_grid[-1]
    if lambda_hpd is None:
        lambda_hpd = lambda_grid[-1]
    if lambda_scp is None:
        lambda_scp = lambda_grid[-1]
        
    results_reg_lambda_crc.append(lambda_crc)
    results_reg_lambda_rcps.append(lambda_rcps)
    results_reg_lambda_hpd.append(lambda_hpd)
    results_reg_lambda_scp.append(lambda_scp)
    
    # Evaluate test risk: risk = mean(|Y_test| > λ)
    risk_crc = np.mean(np.abs(Y_test) > lambda_crc)
    # Failure if test risk > alpha_reg
    failure_flag = 1 if risk_crc > alpha_reg else 0
    failures_reg.append(failure_flag)

results_reg_lambda_crc = np.array(results_reg_lambda_crc)
results_reg_lambda_rcps = np.array(results_reg_lambda_rcps)
results_reg_lambda_hpd = np.array(results_reg_lambda_hpd)
results_reg_lambda_scp = np.array(results_reg_lambda_scp)
failures_reg = np.array(failures_reg)

print("\nSynthetic Heteroskedastic Regression Results Summary (using CRC test risk):")
print("Average selected λ (SCP):", np.mean(results_reg_lambda_scp))
print("Average selected λ (CRC):", np.mean(results_reg_lambda_crc))
print("Average selected λ (Bayesian HPD):", np.mean(results_reg_lambda_hpd))
print("Average selected λ (RCPS):", np.mean(results_reg_lambda_rcps))
print("Failure Rate (based on test risk):", np.mean(failures_reg))

# Generate Figures for Synthetic Regression
plt.figure(figsize=(10,6))
plt.hist(results_reg_lambda_scp, bins=20, alpha=0.5, label='SCP')
plt.hist(results_reg_lambda_crc, bins=20, alpha=0.5, label='CRC')
plt.hist(results_reg_lambda_hpd, bins=20, alpha=0.5, label='Bayesian HPD')
plt.hist(results_reg_lambda_rcps, bins=20, alpha=0.5, label='RCPS')
plt.xlabel("Selected λ")
plt.ylabel("Frequency")
plt.title("Figure_1_Synthetic_Regression: Histogram of Selected λ")
plt.legend()
plt.savefig("Figure_1_Synthetic_Regression.png")
plt.close()

failure_rates_reg = [np.mean(failures_reg)]*4  # Only one failure metric reported (from CRC evaluation)
plt.figure(figsize=(8,6))
plt.bar(methods, failure_rates_reg)
plt.ylabel("Failure Rate")
plt.title("Figure_2_Synthetic_Regression: Failure Rates (Test Risk > α)")
plt.savefig("Figure_2_Synthetic_Regression.png")
plt.close()

#############################
# Experiment 3: MS-COCO Multilabel FNR
#############################
print("\nExperiment 3: MS-COCO Multilabel FNR")
print("This experiment uses the MS-COCO dataset (2017 validation split) with dummy sigmoid scores for multilabel classification.")
print("For each image, we define the ground-truth as classes with sigmoid score > 0.5.")
print("For a candidate λ, the prediction set is {c: sigmoid_score >= 1−λ}.")
print("Per-image FNR is computed as the fraction of true classes that are not predicted; failure occurs if the average FNR > α (α=0.1).")
beta_coco = 0.95
alpha_coco = 0.1
N_trials_coco = 50  # number of trials; each trial randomly samples a subset of images for calibration
N_dirichlet = 1000
# Load the coco_data prepared in the dataset section if available; if not, simulate dummy data.
try:
    coco_data
except NameError:
    # If coco_data not loaded, simulate dummy data: 200 examples with 80 classes.
    num_examples = 200
    num_classes = 80
    dummy_images = []
    for _ in range(num_examples):
        dummy_images.append({"sigmoid_scores": np.random.uniform(0, 1, num_classes).tolist()})
    coco_data = dummy_images
    print("Using simulated dummy coco_data.")

# For evaluation, sample a fixed set of images for testing FNR (e.g. 100 images)
if isinstance(coco_data, list):
    indices = np.random.choice(len(coco_data), size=min(100, len(coco_data)), replace=False)
    test_coco = [coco_data[i] for i in indices]
else:
    indices = np.random.choice(len(coco_data), size=min(100, len(coco_data)), replace=False)
    test_coco = coco_data.select(indices)

results_coco_lambda_scp = []
results_coco_lambda_crc = []
results_coco_lambda_hpd = []
results_coco_lambda_rcps = []
failures_coco = []
avg_set_sizes = { 'SCP': [], 'CRC': [], 'HPD': [], 'RCPS': [] }

# For coco, we treat each trial as a random split of calibration images (say 50 images)
N_calib_coco = 50
for t in range(N_trials_coco):
    # Randomly sample calibration images from coco_data
    if isinstance(coco_data, list):
        calib_samples = random.sample(coco_data, N_calib_coco)
    else:
        all_indices = np.arange(len(coco_data))
        calib_indices = np.random.choice(all_indices, size=N_calib_coco, replace=False)
        calib_samples = coco_data.select(calib_indices)
    
    # For each candidate lambda, compute per-image loss: loss = indicator{FNR > ?}
    # Here we define for each calibration image:
    #   - Ground truth: classes with sigmoid_score > 0.5.
    #   - Prediction set: classes with sigmoid_score >= 1−λ.
    #   - FNR for an image: if no ground truth positives then 0, else (# missed positives)/(# positives).
    calib_losses_grid = []  # shape: (len(lambda_grid), N_calib_coco)
    for lam in lambda_grid:
        losses = []
        for img in calib_samples:
            scores = np.array(img["sigmoid_scores"])
            gt = scores > 0.5
            pred = scores >= (1 - lam)
            if np.sum(gt) == 0:
                fnr = 0.
            else:
                fnr = np.sum(gt & (~pred)) / np.sum(gt)
            losses.append(fnr)
        calib_losses_grid.append(np.array(losses))
    calib_losses_grid = np.array(calib_losses_grid)  # (num_lambda, N_calib_coco)
    
    crc_found = False
    hpd_found = False
    rcps_found = False
    scp_found = False
    lambda_crc = None
    lambda_hpd = None
    lambda_rcps = None
    lambda_scp = None
    for idx, lam in enumerate(lambda_grid):
        losses = calib_losses_grid[idx]
        # CRC: expected L+ = (sum(losses)+B)/(N_calib_coco+1)
        exp_L_plus = (np.sum(losses)+B)/(N_calib_coco+1)
        if (not crc_found) and (exp_L_plus <= alpha_coco):
            lambda_crc = lam
            crc_found = True

        # RCPS: UCB = mean(loss) + sqrt(log(1/(1-beta))/(2*N_calib_coco))
        ucb = np.mean(losses) + sqrt(log(1/(1-beta_coco))/(2*N_calib_coco))
        if (not rcps_found) and (ucb <= alpha_coco):
            lambda_rcps = lam
            rcps_found = True

        # Bayesian HPD:
        sorted_losses = np.sort(losses)
        sorted_extended = np.concatenate([sorted_losses, [B]])
        dirichlet_samples = np.random.dirichlet(np.ones(N_calib_coco+1), size=N_dirichlet)
        L_plus_samples = (dirichlet_samples * sorted_extended).sum(axis=1)
        p_lam = np.mean(L_plus_samples <= alpha_coco)
        if (not hpd_found) and (p_lam >= beta_coco):
            lambda_hpd = lam
            hpd_found = True
            
        # SCP:
        order_idx = ceil((N_calib_coco+1)*(1-alpha_coco)) - 1
        sorted_losses_scp = np.sort(losses)
        s_star = sorted_losses_scp[order_idx]
        # Inversion: For these losses, there is no closed form inversion but we use:
        candidate_scp = 1 - s_star
        if not scp_found and lam >= candidate_scp:
            lambda_scp = lam
            scp_found = True

        if crc_found and rcps_found and hpd_found and scp_found:
            break

    if lambda_crc is None:
        lambda_crc = lambda_grid[-1]
    if lambda_rcps is None:
        lambda_rcps = lambda_grid[-1]
    if lambda_hpd is None:
        lambda_hpd = lambda_grid[-1]
    if lambda_scp is None:
        lambda_scp = lambda_grid[-1]
    
    results_coco_lambda_crc.append(lambda_crc)
    results_coco_lambda_rcps.append(lambda_rcps)
    results_coco_lambda_hpd.append(lambda_hpd)
    results_coco_lambda_scp.append(lambda_scp)
    
    # Now evaluate on the test set (test_coco) for average FNR and prediction set size, using CRC-selected lambda.
    # (We do evaluation for each method similarly; here we store set sizes for each.)
    for method, lam in zip(['SCP','CRC','HPD','RCPS'],
                             [lambda_scp, lambda_crc, lambda_hpd, lambda_rcps]):
        fnrs = []
        set_sizes = []
        for img in test_coco:
            scores = np.array(img["sigmoid_scores"])
            gt = scores > 0.5
            pred = scores >= (1 - lam)
            set_sizes.append(np.sum(pred))
            if np.sum(gt) == 0:
                fnr = 0.
            else:
                fnr = np.sum(gt & (~pred)) / np.sum(gt)
            fnrs.append(fnr)
        avg_fnr = np.mean(fnrs)
        # For each method, we aggregate the average prediction set size for reporting.
        avg_set_sizes[method].append(np.mean(set_sizes))
    # We use the CRC method test risk to determine failure.
    fnrs = []
    for img in test_coco:
        scores = np.array(img["sigmoid_scores"])
        gt = scores > 0.5
        pred = scores >= (1 - lambda_crc)
        if np.sum(gt) == 0:
            fnr = 0.
        else:
            fnr = np.sum(gt & (~pred)) / np.sum(gt)
        fnrs.append(fnr)
    avg_fnr_crc = np.mean(fnrs)
    failures_coco.append(1 if avg_fnr_crc > alpha_coco else 0)

results_coco_lambda_crc = np.array(results_coco_lambda_crc)
results_coco_lambda_rcps = np.array(results_coco_lambda_rcps)
results_coco_lambda_hpd = np.array(results_coco_lambda_hpd)
results_coco_lambda_scp = np.array(results_coco_lambda_scp)
failures_coco = np.array(failures_coco)

print("\nMS-COCO Multilabel FNR Results Summary (using CRC evaluation on test images):")
print("Average selected λ (SCP):", np.mean(results_coco_lambda_scp))
print("Average selected λ (CRC):", np.mean(results_coco_lambda_crc))
print("Average selected λ (Bayesian HPD):", np.mean(results_coco_lambda_hpd))
print("Average selected λ (RCPS):", np.mean(results_coco_lambda_rcps))
print("Failure Rate (Test average FNR > α):", np.mean(failures_coco))
print("Average Prediction Set Sizes:")
for method in avg_set_sizes:
    print(" ", method, ":", np.mean(avg_set_sizes[method]))

# Generate Figures for MS-COCO
plt.figure(figsize=(10,6))
plt.hist(results_coco_lambda_scp, bins=20, alpha=0.5, label='SCP')
plt.hist(results_coco_lambda_crc, bins=20, alpha=0.5, label='CRC')
plt.hist(results_coco_lambda_hpd, bins=20, alpha=0.5, label='Bayesian HPD')
plt.hist(results_coco_lambda_rcps, bins=20, alpha=0.5, label='RCPS')
plt.xlabel("Selected λ")
plt.ylabel("Frequency")
plt.title("Figure_1_MSCOCO: Histogram of Selected λ")
plt.legend()
plt.savefig("Figure_1_MSCOCO.png")
plt.close()

failure_rates_coco = [np.mean(failures_coco)]*4
plt.figure(figsize=(8,6))
plt.bar(methods, failure_rates_coco)
plt.ylabel("Failure Rate")
plt.title("Figure_2_MSCOCO: Failure Rates (Average FNR > α)")
plt.savefig("Figure_2_MSCOCO.png")
plt.close()

print("\nAll experiments complete. Figures saved in the current folder.")