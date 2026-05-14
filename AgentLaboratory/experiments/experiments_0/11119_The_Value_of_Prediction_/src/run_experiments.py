import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, DatasetDict
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# =============================================================================
# NOTE: The following dataset simulation code is assumed to be prepended.
# =============================================================================

# Load an external dataset from HuggingFace (we use ag_news)
external_ds = load_dataset("ag_news", split="train")
df_external = pd.DataFrame(external_ds)
n_external = len(df_external)

# Use the size of the external dataset for our synthetic simulations
np.random.seed(123)
n = n_external

# ----- Gaussian Track Simulation -----
R2_target = 0.7                       # Target R^2 value for Gaussian simulation
rho = np.sqrt(R2_target)              # Induced correlation
sigma = 1.0                           # Standard deviation for Y
Y_gauss = np.random.normal(0, sigma, n)
epsilon_gauss = np.random.normal(0, 1, n)
Yhat_gauss = rho * Y_gauss + np.sqrt(1 - rho**2) * epsilon_gauss
alpha_gauss = 0.75                    # Quantile level for Gaussian track
tYhat_gauss = np.quantile(Yhat_gauss, alpha_gauss)
tY_gauss = np.quantile(Y_gauss, alpha_gauss)

# ----- Log-Normal Track Simulation -----
log_Y = np.random.normal(0, sigma, n)
Y_ln = np.exp(log_Y)
epsilon_ln = np.random.normal(0, 1, n)
log_Yhat = rho * log_Y + np.sqrt(1 - rho**2) * epsilon_ln
Yhat_ln = np.exp(log_Yhat)
gamma = 0.3
mult_noise = np.exp(np.random.normal(0, gamma, n))
Y_ln = Y_ln * mult_noise
beta_ln = 0.2                      # Quantile level for lognormal track (lower tail)
tYhat_ln = np.quantile(Yhat_ln, beta_ln)
tY_ln = np.quantile(Y_ln, beta_ln)

# ----- Combine Simulation Tracks with External Data Details -----
combined_df = pd.DataFrame({
    "id": list(range(2 * n)),
    "Y": np.concatenate([Y_gauss, Y_ln]),
    "Yhat": np.concatenate([Yhat_gauss, Yhat_ln]),
    "track": ["gaussian"] * n + ["lognormal"] * n
})
if "label" in df_external.columns:
    combined_df["external_label"] = list(df_external["label"]) * 2
else:
    combined_df["external_label"] = [None] * (2 * n)

shuffled_indices = np.random.permutation(len(combined_df))
combined_df = combined_df.iloc[shuffled_indices].reset_index(drop=True)

total_samples = len(combined_df)
n_train = int(0.65 * total_samples)
n_val = int(0.20 * total_samples)
n_test = total_samples - n_train - n_val

train_df = combined_df.iloc[:n_train].reset_index(drop=True)
val_df = combined_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
test_df = combined_df.iloc[n_train + n_val:].reset_index(drop=True)

train_dataset = Dataset.from_dict(train_df.to_dict(orient="list"))
val_dataset = Dataset.from_dict(val_df.to_dict(orient="list"))
test_dataset = Dataset.from_dict(test_df.to_dict(orient="list"))

data_splits = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print("Diagnostics:")
print("External dataset size used for simulation:", n)
print("Gaussian track (alpha =", alpha_gauss, ") quantiles -> Yhat:", tYhat_gauss, " Y:", tY_gauss)
print("Lognormal track (beta =", beta_ln, ") quantiles -> Yhat:", tYhat_ln, " Y:", tY_ln)
print("Final dataset splits:")
print("  Train samples:", len(data_splits["train"]))
print("  Validation samples:", len(data_splits["validation"]))
print("  Test samples:", len(data_splits["test"]))

# =============================================================================
# Theoretical Track Computations: V, Local Derivatives, and PAR Grids.
# =============================================================================
print("\n[Theory Track] The following results show simulation-based theoretical computations.")
print("They compute the value function V, its numerical derivatives, and Prediction-Access Ratio (PAR) grids.")

# V(α, β, R2) = Φ2(Φ⁻¹(α), Φ⁻¹(β); ρ)/β, with ρ = sqrt(R2)
def compute_V(alpha, beta, R2):
    rho_val = np.sqrt(R2)
    x = norm.ppf(alpha)
    y = norm.ppf(beta)
    cov = [[1, rho_val], [rho_val, 1]]
    cdf_val = multivariate_normal.cdf([x, y], mean=[0, 0], cov=cov)
    return cdf_val / beta

# Compute base V for given alpha_gauss, beta_ln, and target R2
V_base = compute_V(alpha_gauss, beta_ln, R2_target)
print("Theoretical V(α =", alpha_gauss, ", β =", beta_ln, ", R2 =", R2_target, ") =", V_base)

# Finite-difference derivative with respect to α
delta = 1e-5
V_alpha_plus = compute_V(alpha_gauss + delta, beta_ln, R2_target)
V_alpha_minus = compute_V(alpha_gauss - delta, beta_ln, R2_target)
dV_dalpha_fd = (V_alpha_plus - V_alpha_minus) / (2 * delta)
print("Finite-difference derivative ∂V/∂α at α =", alpha_gauss, "is", dV_dalpha_fd)

# Finite-difference derivative with respect to R2
V_R2_plus = compute_V(alpha_gauss, beta_ln, R2_target + delta)
V_R2_minus = compute_V(alpha_gauss, beta_ln, R2_target - delta)
dV_dR2_fd = (V_R2_plus - V_R2_minus) / (2 * delta)
print("Finite-difference derivative ∂V/∂R2 at R2 =", R2_target, "is", dV_dR2_fd)

# Compute the Prediction-Access Ratio (PAR) for finite improvements Δα and ΔR2 in {0.01, 0.1}
print("\nComputing PAR for selected improvements (Δα and ΔR2 in {0.01, 0.1}):")
par_results = []
for delta_alpha in [0.01, 0.1]:
    for delta_R2 in [0.01, 0.1]:
        V_alpha_inc = compute_V(alpha_gauss + delta_alpha, beta_ln, R2_target)
        V_R2_inc = compute_V(alpha_gauss, beta_ln, R2_target + delta_R2)
        numerator = V_alpha_inc - compute_V(alpha_gauss, beta_ln, R2_target)
        denominator = V_R2_inc - compute_V(alpha_gauss, beta_ln, R2_target)
        PAR = numerator / denominator if denominator != 0 else np.nan
        par_results.append((delta_alpha, delta_R2, PAR))
        print("PAR(Δα =", delta_alpha, ", ΔR2 =", delta_R2, ") =", PAR)

# Create a grid for PAR heatmap over α in [0.5, 0.95] and R2 in [0.1, 0.9] (with fixed β = beta_ln)
alphas = np.linspace(0.5, 0.95, 50)
R2s = np.linspace(0.1, 0.9, 50)
PAR_grid = np.zeros((len(R2s), len(alphas)))

for i, r2_val in enumerate(R2s):
    for j, a in enumerate(alphas):
        num = compute_V(a + 0.01, beta_ln, r2_val) - compute_V(a, beta_ln, r2_val)
        den = compute_V(a, beta_ln, r2_val + 0.01) - compute_V(a, beta_ln, r2_val)
        PAR_grid[i, j] = num / den if den != 0 else np.nan

# =============================================================================
# Empirical Track Experiments: Model Training, Evaluation, and Policy Simulations.
# =============================================================================
print("\n[Empirical Track] The following results demonstrate training complex and simple models, evaluating their performance, conducting policy evaluation, and simulating improved predictions.")

# Prepare training and test data from the combined simulated dataset.
train_X = train_df[['Yhat']].values
train_Y = train_df['Y'].values
val_X = val_df[['Yhat']].values
val_Y = val_df['Y'].values
test_X = test_df[['Yhat']].values
test_Y = test_df['Y'].values

# Train a Gradient Boosting Regressor (Complex Model)
print("\nTraining Gradient Boosting Regressor (Complex Model):")
# Using n_estimators=5000 and early stopping with n_iter_no_change=20
gb_model = GradientBoostingRegressor(n_estimators=5000, n_iter_no_change=20, validation_fraction=0.2, random_state=123)
gb_model.fit(train_X, train_Y)
gb_preds = gb_model.predict(test_X)
r2_gb = r2_score(test_Y, gb_preds)
print("Gradient Boosting Test R²:", r2_gb)

# Train a Decision Tree Regressor (Simple Model)
print("\nTraining Decision Tree Regressor (Simple Model):")
tree_model = DecisionTreeRegressor(max_depth=4, random_state=123)
tree_model.fit(train_X, train_Y)
tree_preds = tree_model.predict(test_X)
r2_tree = r2_score(test_Y, tree_preds)
print("Decision Tree Test R²:", r2_tree)

# Define policy evaluation function
def policy_value(predictions, true_Y, thresh_pred, thresh_true):
    indicator = ((predictions <= thresh_pred) & (true_Y <= thresh_true)).astype(int)
    base = (true_Y <= thresh_true).astype(int)
    # Guard against division by zero
    return indicator.sum() / base.sum() if base.sum() > 0 else np.nan

# Compute thresholds for policy evaluation using quantiles:
tY_true = np.quantile(test_Y, beta_ln)            # Threshold on true Y for β quantile
tYhat_gb = np.quantile(gb_preds, alpha_gauss)      # Threshold on complex model predictions for α quantile
tYhat_tree = np.quantile(tree_preds, alpha_gauss)    # Threshold on simple model predictions for α quantile

V_gb = policy_value(gb_preds, test_Y, tYhat_gb, tY_true)
V_tree = policy_value(tree_preds, test_Y, tYhat_tree, tY_true)
print("\nPolicy Value on Test Set:")
print("Gradient Boosting Model V(α =", alpha_gauss, ", β =", beta_ln, ") =", V_gb)
print("Decision Tree Model V(α =", alpha_gauss, ", β =", beta_ln, ") =", V_tree)

# Simulate improved predictions without retraining for the complex model.
print("\nSimulating improved predictions for Gradient Boosting (Complex Model):")
current_R2 = r2_gb
var_Y = np.var(test_Y)
residuals = test_Y - gb_preds
var_res = np.var(residuals)
for improvement in [0.01, 0.1]:
    target_R2 = min(current_R2 + improvement, 0.99)
    # Derived from: new_R2 = 1 - (1-δ)^2 * Var(error)/Var(Y)
    factor = 1 - np.sqrt((1 - target_R2) / (1 - current_R2))
    adjusted_preds = gb_preds + factor * (test_Y - gb_preds)
    new_R2 = r2_score(test_Y, adjusted_preds)
    tYhat_adjusted = np.quantile(adjusted_preds, alpha_gauss)
    V_adjusted = policy_value(adjusted_preds, test_Y, tYhat_adjusted, tY_true)
    print("For ΔR2 target =", improvement)
    print("  Computed δ factor:", factor)
    print("  New Test R² after adjustment:", new_R2)
    print("  Policy Value V after adjustment:", V_adjusted)

# Special Scenarios:
print("\nSpecial Scenarios:")
# Randomized screening scenario: simulate random predictions by shuffling true Y.
random_preds = np.random.permutation(test_Y)
tYhat_random = np.quantile(random_preds, alpha_gauss)
V_random = policy_value(random_preds, test_Y, tYhat_random, tY_true)
print("Randomized Screening Scenario V:", V_random)

# Near-perfect prediction scenario: nearly perfect predictions with a little noise.
nearperfect_preds = test_Y + np.random.normal(0, 0.01, len(test_Y))
tYhat_nearperfect = np.quantile(nearperfect_preds, alpha_gauss)
V_nearperfect = policy_value(nearperfect_preds, test_Y, tYhat_nearperfect, tY_true)
print("Near-Perfect Prediction Scenario V:", V_nearperfect)

# Capacity Gap Analysis:
print("\nCapacity Gap Analysis:")
# Find minimal Δα* such that the simple model's increase in policy value matches the gap with the complex model.
alpha_grid = np.linspace(alpha_gauss, 0.95, 50)
V_tree_values = [policy_value(tree_preds, test_Y, np.quantile(tree_preds, a), tY_true) for a in alpha_grid]
gap = V_gb - V_tree
delta_alpha_star = np.nan
for i in range(1, len(alpha_grid)):
    if (V_tree_values[i] - V_tree_values[0]) >= gap:
        delta_alpha_star = alpha_grid[i] - alpha_grid[0]
        break
print("Minimal Δα* required for the Decision Tree to match Gradient Boosting gains:", delta_alpha_star)

# Subgroup and Robustness Analysis:
print("\nSubgroup Analysis by External Label:")
unique_labels = np.unique(test_df['external_label'])
for lbl in unique_labels:
    subgroup_idx = (test_df['external_label'] == lbl).values
    if subgroup_idx.sum() == 0:
        continue
    sub_preds = gb_preds[subgroup_idx]
    sub_Y = test_Y[subgroup_idx]
    sub_threshold = np.quantile(sub_Y, beta_ln)
    sub_tYhat = np.quantile(sub_preds, alpha_gauss)
    boot_vals = []
    for _ in range(100):
        indices = np.random.randint(0, len(sub_Y), len(sub_Y))
        boot_val = policy_value(sub_preds[indices], sub_Y[indices], sub_tYhat, sub_threshold)
        boot_vals.append(boot_val)
    boot_vals = np.array(boot_vals)
    ci_lower = np.percentile(boot_vals, 2.5)
    ci_upper = np.percentile(boot_vals, 97.5)
    print("External Label:", lbl, "-> Policy V =", policy_value(sub_preds, sub_Y, sub_tYhat, sub_threshold),
          "with 95% CI = [", ci_lower, ",", ci_upper, "]")

# =============================================================================
# Figures Generation
# =============================================================================
exp_name = "research_exp"

# Figure 1: Screening Policy Curves
print("\nFigure 1: This figure plots the screening policy curves, showing the empirical probability that the model prediction is below the threshold across binned true Y values. It illustrates how screening decisions vary with Y.")
bins = np.linspace(np.min(test_Y), np.max(test_Y), 20)
bin_centers = (bins[:-1] + bins[1:]) / 2
probabilities = []
for i in range(len(bins)-1):
    idx = (test_Y >= bins[i]) & (test_Y < bins[i+1])
    if idx.sum() > 0:
        prob = np.mean(gb_preds[idx] <= tYhat_gb)
    else:
        prob = np.nan
    probabilities.append(prob)
plt.figure()
plt.plot(bin_centers, probabilities, marker='o', linestyle='-')
plt.xlabel("True Y Bins")
plt.ylabel("Probability(Prediction <= Threshold)")
plt.title("Figure_1_" + exp_name + ": Screening Policy Curve (Gradient Boosting)")
plt.grid(True)
plt.savefig("Figure_1_" + exp_name + ".png")
plt.close()
print("Figure_1_" + exp_name + ".png generated successfully.")

# Figure 2: PAR Heatmap
print("Figure 2: This heatmap visualizes the Prediction-Access Ratio (PAR) over a grid of α and R2 values for fixed β = " + str(beta_ln) + ". Contours (in red) indicate cost-adjusted thresholds.")
plt.figure()
im = plt.imshow(PAR_grid, extent=[alphas[0], alphas[-1], R2s[0], R2s[-1]], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(im, label="PAR Value")
plt.xlabel("Quantile Level α")
plt.ylabel("R² Value")
plt.title("Figure_2_" + exp_name + ": PAR Heatmap (β=" + str(beta_ln) + ")")
cs = plt.contour(alphas, R2s, PAR_grid, levels=[1], colors='red')
plt.clabel(cs, inline=True, fontsize=10)
plt.savefig("Figure_2_" + exp_name + ".png")
plt.close()
print("Figure_2_" + exp_name + ".png generated successfully.")

print("\nAll experiments completed successfully.")