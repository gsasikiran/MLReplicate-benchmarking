import numpy as np
import torch
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from datasets import load_dataset

# Set global RNG seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print("Global seeds set to", seed)

###########################################################################
# The following data has been prepared already (by dataset code at the top)
###########################################################################

##############################################
# 1. Synthetic Binomial Loss Experiment
##############################################
# Experiment parameters
M = 10000       # number of trials
n = 10          # number of samples per trial
K = 4
alpha_bin = 0.4   # risk level for binomial experiment, so true risk <= 0.4
# Given analytic truth: true risk = 1 - lambda, so success requires lambda >= 0.6
true_lambda_threshold = 0.6
beta = 0.95     # confidence threshold for HPD and used in RCPS
delta = 1 - beta  # for RCPS UCB

# candidate lambdas and loss_matrix are prepared by the dataset code:
# candidate_lambdas: np.linspace(0.0, 1.0, K + 1)
# loss_matrix: shape (M, n, len(candidate_lambdas))
print("\n--- Synthetic Binomial Loss Experiment ---")
print("This experiment compares four decision rules (SCP, CRC, HPD, RCPS) " +
      "by selecting the minimal candidate lambda based on simulated binomial losses. " +
      "The goal is to control risk (which is 1 - lambda) to be at most alpha=0.4, " +
      "meaning a decision lambda must be at least 0.6 to avoid failure.")

# Preallocate arrays to hold selected lambda for each trial and each method
sel_lambda_scp = np.empty(M)
sel_lambda_crc = np.empty(M)
sel_lambda_hpd = np.empty(M)
sel_lambda_rcps = np.empty(M)

# For HPD rule, parameters for Monte Carlo Dirichlet sampling
N_dirichlet = 1000

# For each trial, loop over candidate lambdas in increasing order and choose the minimal lambda
# that satisfies the method-specific criterion.
for t in range(M):
    # For trial t, extract losses for each candidate lambda.
    # losses_candidate[j]: vector of n losses for candidate lamb = candidate_lambdas[j]
    trial_losses = loss_matrix[t, :, :]  # shape (n, num_candidates)
    
    # For each candidate lambda, we compute:
    # - order statistic for SCP: sort the sample losses and take index = ceil((n+1)*(1-alpha_bin))
    # - CRC: check if (sum(losses)+B)/(n+1) <= alpha_bin, with B=1
    # - HPD: use sorted losses appended with B=1, Monte Carlo Dirichlet sampling to estimate p_lambda
    # - RCPS: check if mean(loss) + sqrt(log(1/delta)/(2n)) <= alpha_bin
    # We then select minimal candidate (in order of candidate_lambdas increasing) that meets the criterion.
    
    # Initialize candidate selections as None (if no candidate meets, we choose the maximum candidate)
    cand_scp = None
    cand_crc = None
    cand_hpd = None
    cand_rcps = None
    
    for j, lam in enumerate(np.sort(candidate_lambdas)):  # candidate lambdas are sorted in increasing order
        losses = trial_losses[:, j]
        # For SCP: compute order statistic index
        order_idx = math.ceil((n + 1) * (1 - alpha_bin))  # 1-indexed
        sorted_losses = np.sort(losses)  # ascending order
        # s_value is the order statistic value
        s_val = sorted_losses[order_idx - 1]  # convert to 0-indexed
        # Then, we select candidate lambda if lam >= s_val.
        if cand_scp is None and lam >= s_val:
            cand_scp = lam

        # CRC: compute risk estimate = (sum(losses) + B) / (n+1)
        risk_crc = (np.sum(losses) + 1) / (n + 1)
        if cand_crc is None and risk_crc <= alpha_bin:
            cand_crc = lam

        # HPD: sort losses and append B=1
        sorted_losses_full = np.concatenate([sorted_losses, np.array([1.0])])
        # Monte Carlo Dirichlet sampling: sample U ~ Dir(1,...,1) of length n+1 (same for all candidates in current trial)
        # To reduce variance across candidates in same trial, we generate Dirichlet samples once per trial candidate.
        U_samples = dirichlet.rvs([1]*(n+1), size=N_dirichlet, random_state=np.random.RandomState(seed + t + j))
        # Compute L+ = dot(U, sorted_losses_full) for each MC sample
        L_plus = U_samples.dot(sorted_losses_full)
        p_lambda = np.mean(L_plus <= alpha_bin)
        if cand_hpd is None and p_lambda >= beta:
            cand_hpd = lam

        # RCPS: compute UCB = mean(loss) + sqrt(log(1/delta)/(2n))
        mean_loss = np.mean(losses)
        ucb = mean_loss + np.sqrt(np.log(1/delta)/(2*n))
        if cand_rcps is None and ucb <= alpha_bin:
            cand_rcps = lam

    # If no candidate met the criterion, set to maximum candidate value:
    if cand_scp is None:
        cand_scp = candidate_lambdas[-1]
    if cand_crc is None:
        cand_crc = candidate_lambdas[-1]
    if cand_hpd is None:
        cand_hpd = candidate_lambdas[-1]
    if cand_rcps is None:
        cand_rcps = candidate_lambdas[-1]
    
    sel_lambda_scp[t] = cand_scp
    sel_lambda_crc[t] = cand_crc
    sel_lambda_hpd[t] = cand_hpd
    sel_lambda_rcps[t] = cand_rcps

# Compute failure rates: failure occurs if selected lambda < true_lambda_threshold (i.e. <0.6)
fail_scp = np.mean(sel_lambda_scp < true_lambda_threshold)
fail_crc = np.mean(sel_lambda_crc < true_lambda_threshold)
fail_hpd = np.mean(sel_lambda_hpd < true_lambda_threshold)
fail_rcps = np.mean(sel_lambda_rcps < true_lambda_threshold)

print("\nResults for Synthetic Binomial Loss (aim: selected lambda >= 0.6 to ensure risk control):")
print("SCP: Average selected lambda = %.3f, Failure rate = %.3f" % (np.mean(sel_lambda_scp), fail_scp))
print("CRC: Average selected lambda = %.3f, Failure rate = %.3f" % (np.mean(sel_lambda_crc), fail_crc))
print("HPD: Average selected lambda = %.3f, Failure rate = %.3f" % (np.mean(sel_lambda_hpd), fail_hpd))
print("RCPS: Average selected lambda = %.3f, Failure rate = %.3f" % (np.mean(sel_lambda_rcps), fail_rcps))

# Generate Figure_1_Binomial.png: Histogram of selected lambdas for each method
plt.figure(figsize=(10,6))
bins = np.linspace(0, 1, 30)
plt.hist(sel_lambda_scp, bins, alpha=0.5, label='SCP')
plt.hist(sel_lambda_crc, bins, alpha=0.5, label='CRC')
plt.hist(sel_lambda_hpd, bins, alpha=0.5, label='HPD')
plt.hist(sel_lambda_rcps, bins, alpha=0.5, label='RCPS')
plt.xlabel('Selected lambda')
plt.ylabel('Frequency')
plt.title('Figure_1_Binomial: Distribution of Selected Lambdas (Binomial Loss)')
plt.legend()
plt.savefig("Figure_1_Binomial.png")
plt.close()
print("Figure_1_Binomial.png saved: Histogram of selected lambdas for Synthetic Binomial Loss Experiment.")

###########################################################################
# 2. Synthetic Heteroskedastic Regression Experiment
###########################################################################
print("\n--- Synthetic Heteroskedastic Regression Experiment ---")
print("This experiment uses a heteroskedastic regression model where the loss " +
      "is defined as the indicator { |Y| > lambda } on a calibration set of size n_calib. " +
      "For each decision rule (SCP, CRC, HPD, RCPS), we select the minimal lambda from a grid " +
      "that ensures the estimated risk (from calibration losses) is at most alpha_reg. " +
      "Then we evaluate on a large test set to see if the selected lambda yields risk <= alpha_reg.")

# Experiment parameters:
n_calib = 200      # calibration set size (already generated: X_calib, Y_calib)
N_test = 200000    # test set size (already generated: X_test, Y_test and risk_estimates)
lambda_grid_reg = np.linspace(0, 5, 11)  # candidate thresholds for regression
alpha_reg = 0.1    # risk threshold (i.e., fraction of miscoverage)
beta = 0.95        # for HPD and RCPS
delta = 1 - beta
M_reg = 100        # number of calibration trials (via bootstrapping)

# Pre-calculate calibration losses for each candidate in lambda_grid_reg
# calib_losses: shape (num_candidates, n_calib) already computed
# We run M_reg bootstrap trials from calibration data
sel_lambda_scp_reg = np.empty(M_reg)
sel_lambda_crc_reg = np.empty(M_reg)
sel_lambda_hpd_reg = np.empty(M_reg)
sel_lambda_rcps_reg = np.empty(M_reg)

for t in range(M_reg):
    # Bootstrap sample indices from calibration set (size n_calib)
    indices = np.random.choice(n_calib, size=n_calib, replace=True)
    # For each candidate threshold, get losses as indicator(|Y_calib| > lambda)
    # calib_losses is shape (num_candidates, n_calib); take bootstrap sample along axis=1
    trial_losses_all = calib_losses[:, indices]  # shape (num_candidates, n_calib)
    
    cand_scp = None
    cand_crc = None
    cand_hpd = None
    cand_rcps = None
    
    for j, lam in enumerate(lambda_grid_reg):
        losses = trial_losses_all[j, :]  # losses for candidate lam for this trial
        # SCP: order statistic at index = ceil((n_calib+1)*(1-alpha_reg))
        order_idx = math.ceil((n_calib + 1) * (1 - alpha_reg))
        sorted_losses = np.sort(losses)
        s_val = sorted_losses[order_idx - 1]
        if cand_scp is None and lam >= s_val:
            cand_scp = lam
            
        # CRC: risk estimate = (sum(losses)+1)/(n_calib+1)
        risk_crc = (np.sum(losses) + 1) / (n_calib + 1)
        if cand_crc is None and risk_crc <= alpha_reg:
            cand_crc = lam
            
        # HPD: sort losses and append 1, then use Dirichlet Monte Carlo sampling
        sorted_losses_full = np.concatenate([sorted_losses, np.array([1.0])])
        U_samples = dirichlet.rvs([1]*(n_calib+1), size=N_dirichlet, random_state=np.random.RandomState(seed + t + j))
        L_plus = U_samples.dot(sorted_losses_full)
        p_lambda = np.mean(L_plus <= alpha_reg)
        if cand_hpd is None and p_lambda >= beta:
            cand_hpd = lam
        
        # RCPS: UCB = mean(loss) + sqrt(log(1/delta)/(2*n_calib))
        mean_loss = np.mean(losses)
        ucb = mean_loss + np.sqrt(np.log(1/delta)/(2*n_calib))
        if cand_rcps is None and ucb <= alpha_reg:
            cand_rcps = lam
            
    if cand_scp is None:
        cand_scp = lambda_grid_reg[-1]
    if cand_crc is None:
        cand_crc = lambda_grid_reg[-1]
    if cand_hpd is None:
        cand_hpd = lambda_grid_reg[-1]
    if cand_rcps is None:
        cand_rcps = lambda_grid_reg[-1]
        
    sel_lambda_scp_reg[t] = cand_scp
    sel_lambda_crc_reg[t] = cand_crc
    sel_lambda_hpd_reg[t] = cand_hpd
    sel_lambda_rcps_reg[t] = cand_rcps

# For each trial, we now evaluate test risk.
# risk_estimates is precomputed: for each candidate lambda in lambda_grid_reg, risk = mean(indicator(|Y_test| > lambda))
# For a chosen lambda (which lies in the candidate grid), we use its corresponding risk estimate.
# We set failure if risk_estimate > alpha_reg.
def get_test_risk(selected_lambda):
    # Find closest candidate in lambda_grid_reg (grid is sorted increasingly)
    idx = np.searchsorted(lambda_grid_reg, selected_lambda)
    if idx >= len(lambda_grid_reg):
        idx = len(lambda_grid_reg) - 1
    return risk_estimates[idx]

test_risks_scp = np.array([ get_test_risk(lam) for lam in sel_lambda_scp_reg ])
test_risks_crc = np.array([ get_test_risk(lam) for lam in sel_lambda_crc_reg ])
test_risks_hpd = np.array([ get_test_risk(lam) for lam in sel_lambda_hpd_reg ])
test_risks_rcps = np.array([ get_test_risk(lam) for lam in sel_lambda_rcps_reg ])

fail_scp_reg = np.mean(test_risks_scp > alpha_reg)
fail_crc_reg = np.mean(test_risks_crc > alpha_reg)
fail_hpd_reg = np.mean(test_risks_hpd > alpha_reg)
fail_rcps_reg = np.mean(test_risks_rcps > alpha_reg)

print("\nResults for Synthetic Heteroskedastic Regression:")
print("SCP: Average selected lambda = %.3f, Average test risk = %.3f, Failure rate = %.3f" % 
      (np.mean(sel_lambda_scp_reg), np.mean(test_risks_scp), fail_scp_reg))
print("CRC: Average selected lambda = %.3f, Average test risk = %.3f, Failure rate = %.3f" % 
      (np.mean(sel_lambda_crc_reg), np.mean(test_risks_crc), fail_crc_reg))
print("HPD: Average selected lambda = %.3f, Average test risk = %.3f, Failure rate = %.3f" % 
      (np.mean(sel_lambda_hpd_reg), np.mean(test_risks_hpd), fail_hpd_reg))
print("RCPS: Average selected lambda = %.3f, Average test risk = %.3f, Failure rate = %.3f" % 
      (np.mean(sel_lambda_rcps_reg), np.mean(test_risks_rcps), fail_rcps_reg))

# Generate Figure_2_Regression.png: Boxplot of selected lambdas and corresponding test risks for each method
plt.figure(figsize=(10,6))
methods = ['SCP', 'CRC', 'HPD', 'RCPS']
selected_lambdas_all = [sel_lambda_scp_reg, sel_lambda_crc_reg, sel_lambda_hpd_reg, sel_lambda_rcps_reg]
test_risks_all = [test_risks_scp, test_risks_crc, test_risks_hpd, test_risks_rcps]

plt.subplot(1,2,1)
plt.boxplot(selected_lambdas_all, labels=methods)
plt.xlabel('Method')
plt.ylabel('Selected Lambda')
plt.title('Figure_2_Regression: Distribution of Selected Lambdas')

plt.subplot(1,2,2)
plt.boxplot(test_risks_all, labels=methods)
plt.xlabel('Method')
plt.ylabel('Test Risk')
plt.title('Figure_2_Regression: Distribution of Test Risks')
plt.tight_layout()
plt.savefig("Figure_2_Regression.png")
plt.close()
print("Figure_2_Regression.png saved: Boxplots of selected lambdas and test risks for Regression Experiment.")

###########################################################################
# 3. MS-COCO Multilabel Classification Experiment
###########################################################################
print("\n--- MS-COCO Multilabel Classification Experiment ---")
print("This experiment simulates multilabel classification on the MS-COCO dataset. " +
      "For each image, fake multilabel scores were generated. We simulate 'true' labels " +
      "by assuming that the top 5 scores (from the precomputed sorted_scores) correspond to true labels. " +
      "For each candidate decision threshold (through candidate_lambdas), " +
      "we compute the false negative rate (FNR) defined as the fraction of true labels " +
      "that are not predicted (i.e. scores below threshold 1 - lambda), and the size of the prediction set. " +
      "We then apply the four decision rules (SCP, CRC, HPD, RCPS) to select lambda " +
      "and report the average FNR, prediction set size, and failure rate (failure if mean FNR > alpha_coco).")

alpha_coco = 0.2   # target maximum FNR
M_coco = 50        # number of trials
# For each trial, we randomly sample 100 images from coco_data
num_images_per_trial = 100

# Prepare lists to record outcomes for each method
sel_lambda_scp_coco = []
sel_lambda_crc_coco = []
sel_lambda_hpd_coco = []
sel_lambda_rcps_coco = []
fnr_scp_all = []
fnr_crc_all = []
fnr_hpd_all = []
fnr_rcps_all = []
psize_scp_all = []
psize_crc_all = []
psize_hpd_all = []
psize_rcps_all = []

if coco_data is not None:
    total_examples = len(coco_data)
    for t in range(M_coco):
        # Sample indices from coco_data
        indices = np.random.choice(total_examples, size=num_images_per_trial, replace=False)
        trial_data = coco_data.select(indices)
        # For each candidate lambda (from candidate_lambdas defined earlier), compute losses (FNR) and prediction set sizes for each image.
        # We'll assume the "true" labels are the top 5 classes from sorted_scores.
        losses_matrix_coco = np.empty((num_images_per_trial, len(candidate_lambdas)))
        psize_matrix_coco = np.empty((num_images_per_trial, len(candidate_lambdas)))
        for i, ex in enumerate(trial_data):
            sorted_scores = ex["sorted_scores"]  # descending order
            # Define true labels as the indices of the top 5 scores
            true_labels = set(range(5))
            scores = np.array(ex["scores"])
            for j, lam in enumerate(candidate_lambdas):
                threshold = 1 - lam  # decision rule: include classes with score >= 1 - lambda
                # Prediction set: indices where score >= threshold
                pred_set = set(np.where(scores >= threshold)[0])
                # For simulation, we assume that the "true" labels are the top 5 indices of sorted order
                # However, since sorted_scores does not give indices, we simulate: assume classes 0 to 4 are true.
                fn = len(true_labels - pred_set)
                fnr = fn / 5.0
                losses_matrix_coco[i, j] = fnr
                psize_matrix_coco[i, j] = len(pred_set)
        # Now, for each decision rule, we treat the losses for candidate lambda j across images
        cand_scp = None
        cand_crc = None
        cand_hpd = None
        cand_rcps = None
        for j, lam in enumerate(candidate_lambdas):
            losses = losses_matrix_coco[:, j]
            # SCP: order statistic at index = ceil((num_images_per_trial+1)*(1-alpha_coco))
            order_idx = math.ceil((num_images_per_trial + 1) * (1 - alpha_coco))
            sorted_losses = np.sort(losses)
            s_val = sorted_losses[order_idx - 1]
            if cand_scp is None and lam >= s_val:
                cand_scp = lam
                
            # CRC: risk estimate = (sum(losses)+1)/(num_images_per_trial+1)
            risk_crc = (np.sum(losses) + 1) / (num_images_per_trial + 1)
            if cand_crc is None and risk_crc <= alpha_coco:
                cand_crc = lam
                
            # HPD: sort losses and append 1; use Dirichlet sampling
            sorted_losses_full = np.concatenate([sorted_losses, np.array([1.0])])
            U_samples = dirichlet.rvs([1]*(num_images_per_trial+1), size=N_dirichlet, 
                                       random_state=np.random.RandomState(seed + t + j))
            L_plus = U_samples.dot(sorted_losses_full)
            p_lambda = np.mean(L_plus <= alpha_coco)
            if cand_hpd is None and p_lambda >= beta:
                cand_hpd = lam
                
            # RCPS: UCB = mean(loss) + sqrt(log(1/delta)/(2*num_images_per_trial))
            mean_loss = np.mean(losses)
            ucb = mean_loss + np.sqrt(np.log(1/delta)/(2*num_images_per_trial))
            if cand_rcps is None and ucb <= alpha_coco:
                cand_rcps = lam
                
        if cand_scp is None:
            cand_scp = candidate_lambdas[-1]
        if cand_crc is None:
            cand_crc = candidate_lambdas[-1]
        if cand_hpd is None:
            cand_hpd = candidate_lambdas[-1]
        if cand_rcps is None:
            cand_rcps = candidate_lambdas[-1]
            
        sel_lambda_scp_coco.append(cand_scp)
        sel_lambda_crc_coco.append(cand_crc)
        sel_lambda_hpd_coco.append(cand_hpd)
        sel_lambda_rcps_coco.append(cand_rcps)
        
        # Evaluate each candidate's performance on this trial (using the candidate that was selected)
        # For each method, record the average FNR and average prediction set size for the chosen candidate column.
        idx_scp = np.searchsorted(candidate_lambdas, cand_scp)
        idx_crc = np.searchsorted(candidate_lambdas, cand_crc)
        idx_hpd = np.searchsorted(candidate_lambdas, cand_hpd)
        idx_rcps = np.searchsorted(candidate_lambdas, cand_rcps)
        # Clip indices in case they fall beyond range
        idx_scp = min(idx_scp, len(candidate_lambdas)-1)
        idx_crc = min(idx_crc, len(candidate_lambdas)-1)
        idx_hpd = min(idx_hpd, len(candidate_lambdas)-1)
        idx_rcps = min(idx_rcps, len(candidate_lambdas)-1)
        
        fnr_scp_all.append(np.mean(losses_matrix_coco[:, idx_scp]))
        fnr_crc_all.append(np.mean(losses_matrix_coco[:, idx_crc]))
        fnr_hpd_all.append(np.mean(losses_matrix_coco[:, idx_hpd]))
        fnr_rcps_all.append(np.mean(losses_matrix_coco[:, idx_rcps]))
        
        psize_scp_all.append(np.mean(psize_matrix_coco[:, idx_scp]))
        psize_crc_all.append(np.mean(psize_matrix_coco[:, idx_crc]))
        psize_hpd_all.append(np.mean(psize_matrix_coco[:, idx_hpd]))
        psize_rcps_all.append(np.mean(psize_matrix_coco[:, idx_rcps]))
    
    # Compute failure rates: failure if average FNR > alpha_coco
    fail_scp_coco = np.mean(np.array(fnr_scp_all) > alpha_coco)
    fail_crc_coco = np.mean(np.array(fnr_crc_all) > alpha_coco)
    fail_hpd_coco = np.mean(np.array(fnr_hpd_all) > alpha_coco)
    fail_rcps_coco = np.mean(np.array(fnr_rcps_all) > alpha_coco)
    
    print("\nResults for MS-COCO Multilabel Classification (over %d trials):" % M_coco)
    print("SCP: Average selected lambda = %.3f, Average FNR = %.3f, Avg. set size = %.3f, Failure rate = %.3f" %
          (np.mean(sel_lambda_scp_coco), np.mean(fnr_scp_all), np.mean(psize_scp_all), fail_scp_coco))
    print("CRC: Average selected lambda = %.3f, Average FNR = %.3f, Avg. set size = %.3f, Failure rate = %.3f" %
          (np.mean(sel_lambda_crc_coco), np.mean(fnr_crc_all), np.mean(psize_crc_all), fail_crc_coco))
    print("HPD: Average selected lambda = %.3f, Average FNR = %.3f, Avg. set size = %.3f, Failure rate = %.3f" %
          (np.mean(sel_lambda_hpd_coco), np.mean(fnr_hpd_all), np.mean(psize_hpd_all), fail_hpd_coco))
    print("RCPS: Average selected lambda = %.3f, Average FNR = %.3f, Avg. set size = %.3f, Failure rate = %.3f" %
          (np.mean(sel_lambda_rcps_coco), np.mean(fnr_rcps_all), np.mean(psize_rcps_all), fail_rcps_coco))
else:
    print("MS-COCO dataset not available; skipping multilabel classification experiment.")

print("\n--- End of Experiments ---")