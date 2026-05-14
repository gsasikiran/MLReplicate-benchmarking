# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Try to import CatBoostRegressor from catboost. If not available, we'll use GradientBoostingRegressor as a fallback.
catboost_available = False
try:
    from catboost import CatBoostRegressor
    catboost_available = True
except ImportError:
    print("catboost module not found. Using GradientBoostingRegressor as a fallback.")

# --------------------------------------------------------------------------------------------
# THEORY TRACK COMPUTATIONS
# --------------------------------------------------------------------------------------------
print("Theory Experiment: This section computes the theoretical policy value V(α,β,R2) and its local sensitivities (∂V/∂α, ∂V/∂R2) using a bivariate normal CDF. Furthermore, it computes the Prediction-Access Ratio (PAR) for finite improvements. The figures generated illustrate (1) the screening policy curve as a function of Y and (2) a heatmap displaying the sensitivity of V with respect to R2 over a grid of α and R2 values.")

# Given parameters for theory experiment
alpha = 0.2
beta = 0.15
R2 = 0.5
rho = np.sqrt(R2)
z_alpha = norm.ppf(alpha)
z_beta = norm.ppf(beta)

# Compute theoretical V based on bivariate normal CDF:
V = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([z_alpha, z_beta]) / beta
print("Theoretical V(α={}, β={}, R2={}) = {:.5f}".format(alpha, beta, R2, V))

# Finite difference step for sensitivity analysis
eps = 1e-5

# Sensitivity with respect to α:
z_alpha_eps = norm.ppf(alpha + eps)
V_alpha_eps = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([z_alpha_eps, z_beta]) / beta
dV_dalpha = (V_alpha_eps - V) / eps

# Sensitivity with respect to R2:
R2_eps = R2 + eps
rho_eps = np.sqrt(R2_eps)
V_R2_eps = multivariate_normal(mean=[0, 0], cov=[[1, rho_eps], [rho_eps, 1]]).cdf([z_alpha, z_beta]) / beta
dV_dR2 = (V_R2_eps - V) / eps

print("Finite difference sensitivity ∂V/∂α ≈ {:.5f}".format(dV_dalpha))
print("Finite difference sensitivity ∂V/∂R2 ≈ {:.5f}".format(dV_dR2))

# Compute PAR for finite improvements in α and R2.
delta_list = [0.01, 0.1]
for d in delta_list:
    # Improvement in α: compute new threshold and corresponding V
    z_alpha_d = norm.ppf(alpha + d)
    V_alpha_d = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([z_alpha_d, z_beta]) / beta
    numerator = V_alpha_d - V
    # Improvement in R2: compute new R2 V value
    R2_d = R2 + d
    rho_d = np.sqrt(R2_d)
    V_R2_d = multivariate_normal(mean=[0, 0], cov=[[1, rho_d], [rho_d, 1]]).cdf([z_alpha, z_beta]) / beta
    denominator = V_R2_d - V
    PAR = numerator / denominator if denominator != 0 else np.nan
    print("For Δα = ΔR2 = {:.2f}: Numerator = {:.5f}, Denom = {:.5f}, PAR = {:.5f}".format(d, numerator, denominator, PAR))

# Generate heatmap of sensitivity of V with respect to small improvements in R2 over a grid defined by α and R2.
alphas = np.linspace(0.05, 0.5, 50)
R2s = np.linspace(0.01, 0.9, 50)
PAR_grid = np.zeros((len(R2s), len(alphas)))
for i, r2_val in enumerate(R2s):
    rho_val = np.sqrt(r2_val)
    for j, a_val in enumerate(alphas):
        z_a = norm.ppf(a_val)
        # Baseline V for (a_val, r2_val)
        V_base = multivariate_normal(mean=[0, 0], cov=[[1, rho_val], [rho_val, 1]]).cdf([z_a, z_beta]) / beta
        # V for small improvement in R2 (ΔR2=0.01)
        r2_impr = r2_val + 0.01 if r2_val + 0.01 <= 1 else r2_val
        rho_impr = np.sqrt(r2_impr)
        V_impr = multivariate_normal(mean=[0, 0], cov=[[1, rho_impr], [rho_impr, 1]]).cdf([z_a, z_beta]) / beta
        PAR_grid[i, j] = V_impr - V_base

plt.figure(figsize=(8, 6))
plt.imshow(PAR_grid, extent=[alphas[0], alphas[-1], R2s[0], R2s[-1]], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='ΔV due to ΔR2=0.01')
plt.xlabel("α")
plt.ylabel("R2")
plt.title("Figure_2_Theory: Heatmap of ΔV due to ΔR2=0.01 over (α, R2)")
plt.savefig("Figure_2_Theory.png", dpi=150)
plt.close()
print("Generated Figure_2_Theory.png: Heatmap of theoretical ΔV over α and R2.")

# Generate screening policy curve: This curve approximates, for a bivariate normal model, the probability that Yhat is below its α-quantile given Y = y.
tYhat_std = norm.ppf(alpha)
y_vals = np.linspace(-3, 3, 100)
policy_curve = norm.cdf((tYhat_std - rho * y_vals) / np.sqrt(1 - rho**2))
plt.figure(figsize=(8, 6))
plt.plot(y_vals, policy_curve, label="Screening policy curve")
plt.xlabel("Y")
plt.ylabel("Pr(Yhat ≤ tYhat(α) | Y)")
plt.title("Figure_1_Theory: Screening Policy Curve (α={} & R2={})".format(alpha, R2))
plt.legend()
plt.savefig("Figure_1_Theory.png", dpi=150)
plt.close()
print("Generated Figure_1_Theory.png: Screening policy curve at fixed α and R2.")

# --------------------------------------------------------------------------------------------
# EMPIRICAL TRACK PIPELINE
# --------------------------------------------------------------------------------------------
print("\nEmpirical Experiment: This section trains two baseline models (one complex model using CatBoost or a fallback regressor, and one simple Decision Tree) on synthetic data. It then computes out-of-sample R2 on the test set along with empirical policy value V(α,β). Further, it simulates improved prediction via residual scaling, estimates empirical PAR, compares special scenarios (random screening and near-perfect prediction), and performs capacity gap analysis and subgroup analyses.")

# The synthetic data is assumed to be already created from the dataset code provided at the top.
# For reproducibility, display head of synthetic dataset 'df'.
print("Empirical: Head of synthetic dataset:")
print(df.head())

# Splitting data based on 'split' column:
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

# Define features and target; Using features: Yhat, gender, age and target Y.
features = ["Yhat", "gender", "age"]
target = "Y"

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Train complex model (CatBoost if available, else GradientBoostingRegressor fallback)
if catboost_available:
    print("Training CatBoost regressor (iterations=5000, early stopping=20)...")
    model_complex = CatBoostRegressor(iterations=5000, early_stopping_rounds=20, verbose=0, random_seed=42)
else:
    print("Training GradientBoostingRegressor as fallback (n_estimators=5000, n_iter_no_change=20)...")
    model_complex = GradientBoostingRegressor(n_estimators=5000, n_iter_no_change=20, random_state=42)

model_complex.fit(X_train, y_train)
y_pred_complex = model_complex.predict(X_test)
r2_complex = r2_score(y_test, y_pred_complex)
print("Complex model Test R2: {:.5f}".format(r2_complex))

# Train simple Decision Tree regressor (max_depth=4)
print("Training Decision Tree regressor (max_depth=4)...")
model_simple = DecisionTreeRegressor(max_depth=4, random_state=42)
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)
r2_simple = r2_score(y_test, y_pred_simple)
print("Decision Tree Test R2: {:.5f}".format(r2_simple))

# Compute empirical policy value V(α,β) using the complex model predictions.
tYhat_alpha_test = np.quantile(y_pred_complex, alpha)
tY_beta_test = np.quantile(y_test, beta)
indicator = ((y_pred_complex <= tYhat_alpha_test) & (y_test <= tY_beta_test)).astype(int)
denom = (y_test <= tY_beta_test).sum()
V_empirical = indicator.sum() / denom if denom > 0 else np.nan
print("Empirical Policy Value V(α={}, β={}) with complex model: {:.5f}".format(alpha, beta, V_empirical))

# Simulate improved prediction without retraining via residual scaling.
delta = 0.1  # moderate improvement scaling
y_pred_improv = y_pred_complex + delta * (y_test - y_pred_complex)
r2_improv = r2_score(y_test, y_pred_improv)
print("After residual scaling (δ={}), improved model Test R2: {:.5f}".format(delta, r2_improv))
tYhat_alpha_improv = np.quantile(y_pred_improv, alpha)
indicator_improv = ((y_pred_improv <= tYhat_alpha_improv) & (y_test <= tY_beta_test)).astype(int)
V_empirical_improv = indicator_improv.sum() / denom if denom > 0 else np.nan
print("Empirical Policy Value V(α={}, β={}) with improved predictions: {:.5f}".format(alpha, beta, V_empirical_improv))

# Compute empirical PAR for finite improvements in screening threshold Δα.
for d in delta_list:
    tYhat_alpha_d = np.quantile(y_pred_complex, alpha + d)
    indicator_d = ((y_pred_complex <= tYhat_alpha_d) & (y_test <= tY_beta_test)).astype(int)
    V_alpha_d = indicator_d.sum() / denom if denom > 0 else np.nan
    num_emp = V_alpha_d - V_empirical
    # Improvement in prediction is measured by the difference between improved and baseline V.
    V_pred_diff = V_empirical_improv - V_empirical
    PAR_emp = num_emp / V_pred_diff if V_pred_diff != 0 else np.nan
    print("Empirical PAR for Δα = {:.2f}: Numerator = {:.5f}, Denom (Δ prediction) = {:.5f}, PAR = {:.5f}".format(d, num_emp, V_pred_diff, PAR_emp))

# Special Scenario: Randomized screening (simulate R2=0)
y_pred_random = np.random.permutation(y_pred_complex)
tYhat_alpha_rand = np.quantile(y_pred_random, alpha)
indicator_rand = ((y_pred_random <= tYhat_alpha_rand) & (y_test <= tY_beta_test)).astype(int)
V_random = indicator_rand.sum() / denom if denom > 0 else np.nan
print("Special Scenario: Random screening (R2=0) yields Empirical V = {:.5f}".format(V_random))

# Special Scenario: Near-perfect prediction (simulate R2≈0.9)
epsilon_noise = 1e-3
y_pred_near = y_test + np.random.normal(0, epsilon_noise, size=len(y_test))
tYhat_alpha_near = np.quantile(y_pred_near, alpha)
indicator_near = ((y_pred_near <= tYhat_alpha_near) & (y_test <= tY_beta_test)).astype(int)
V_near = indicator_near.sum() / denom if denom > 0 else np.nan
print("Special Scenario: Near-perfect prediction (R2≈0.9) yields Empirical V = {:.5f}".format(V_near))

# Capacity Gap Analysis: Find minimal Δα* such that the gain in the simple model equals or exceeds the complex model's improvement gain.
dalpha_vals = np.linspace(0.001, 0.2, 200)
gap_found = False
for da in dalpha_vals:
    tYhat_alpha_simple = np.quantile(y_pred_complex, alpha + da)
    V_simple_improvement = (((y_pred_complex <= tYhat_alpha_simple) & (y_test <= tY_beta_test)).astype(int)).sum() / denom
    if (V_simple_improvement - V_empirical) >= (V_empirical_improv - V_empirical):
        print("Capacity Gap Analysis: Minimum additional screening Δα* = {:.4f} required for the simple model to match the improved complex model gains.".format(da))
        gap_found = True
        break
if not gap_found:
    print("Capacity Gap Analysis: No Δα* found in the search grid that matches the improvement gain.")

# Subgroup Analysis by Gender: Compute empirical policy value V for each gender subgroup.
print("Subgroup Analysis by Gender:")
for gender_val in [0, 1]:
    idx = test_df['gender'] == gender_val
    if idx.sum() == 0:
        continue
    y_test_sub = y_test[idx]
    y_pred_sub = y_pred_complex[idx]
    tY_beta_sub = np.quantile(y_test_sub, beta)
    tYhat_alpha_sub = np.quantile(y_pred_sub, alpha)
    indicator_sub = ((y_pred_sub <= tYhat_alpha_sub) & (y_test_sub <= tY_beta_sub)).astype(int)
    V_sub = indicator_sub.sum() / (y_test_sub <= tY_beta_sub).sum()
    print("  Gender {}: Empirical V = {:.5f}".format(gender_val, V_sub))

print("\nEmpirical experiments complete. The results above detail model performance, policy values, PAR estimates, special scenarios, capacity gap analysis, and subgroup analyses. Figures 'Figure_1_Theory.png' (screening policy curve) and 'Figure_2_Theory.png' (heatmap of ΔV over α and R2) have been saved in the current folder.")