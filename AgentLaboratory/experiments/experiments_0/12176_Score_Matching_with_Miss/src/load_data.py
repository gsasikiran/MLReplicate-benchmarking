import numpy as np
import pandas as pd
from datasets import load_dataset

# ----- Section 1: Synthetic Gaussian Data (10-D) -----
np.random.seed(999)
n_samples = 800
dim = 10
# Create covariance matrix using eigen-decomposition with eigenvalues in [1.0, 2.0]
eig_vals = np.random.uniform(1.0, 2.0, dim)
rand_matrix = np.random.randn(dim, dim)
Q, _ = np.linalg.qr(rand_matrix)
covariance = Q @ np.diag(eig_vals) @ Q.T
mean_vector = np.full(dim, 0.5)
gaussian_data = np.random.multivariate_normal(mean_vector, covariance, size=n_samples)
# Impose strong dependency: last dimension = 0.5 * first dimension + 0.5 * noise (variance from col 0)
var_first = np.var(gaussian_data[:, 0])
noise = np.random.normal(0, np.sqrt(var_first), size=n_samples)
gaussian_data[:, -1] = 0.5 * gaussian_data[:, 0] + 0.5 * noise

# ----- Section 2: Truncation and Boundary-Aware Weights -----
# Compute 10th percentile thresholds for dimensions 0,1,2
quantile_10 = np.percentile(gaussian_data[:, :3], 10, axis=0)
# Smooth proxy weights: if value below threshold, weight = exp( - (threshold - value)^2), otherwise 1.
smooth_weights = np.ones_like(gaussian_data[:, :3])
for j in range(3):
    for i in range(n_samples):
        if gaussian_data[i, j] < quantile_10[j]:
            diff = quantile_10[j] - gaussian_data[i, j]
            smooth_weights[i, j] = np.exp(-diff**2)

# ----- Section 3: MCAR Missingness on Gaussian Data -----
p_missing = 0.3
mask_gaussian = np.random.rand(n_samples, dim) > p_missing
gaussian_masked = gaussian_data.copy().astype(float)
gaussian_masked[~mask_gaussian] = np.nan

print("Synthetic Gaussian Data:")
print(" Shape:", gaussian_data.shape)
print(" 10th percentile thresholds (dims 0-2):", quantile_10)
print(" Smooth weights sample (first 5 rows, dims 0-2):\n", smooth_weights[:5])
print(" Missingness mask sample (first 5 rows):\n", mask_gaussian[:5])
print(" First 5 rows of masked data:\n", gaussian_masked[:5])

# ----- Section 4: Graph Structure - Star Graph (hub at index 0) -----
star_adj_mat = np.zeros((dim, dim))
for j in range(1, dim):
    star_adj_mat[0, j] = 1
    star_adj_mat[j, 0] = 1
print("\nStar Graph Adjacency Matrix:")
print(star_adj_mat)

# ----- Section 5: Synthetic ICA-like Data via Rejection Sampling -----
np.random.seed(2021)
ica_sample_target = 250
ica_dim = 10
theta = 0.08
# Maximum density approximated when all |x|=1; number of pairs = dim*(dim-1)/2.
n_pairs = (ica_dim * (ica_dim - 1)) // 2
max_density = np.exp(theta * n_pairs)
ica_samples = []
attempts = 0
while len(ica_samples) < ica_sample_target and attempts < 150000:
    candidate = np.random.uniform(-1, 1, ica_dim)
    s = 0
    for i in range(ica_dim):
        for j in range(i+1, ica_dim):
            s += theta * candidate[i]**2 * candidate[j]**2
    density = np.exp(s)
    if np.random.rand() < density / max_density:
        ica_samples.append(candidate)
    attempts += 1
ica_data = np.array(ica_samples)
print("\nICA-like Data via Rejection Sampling. Shape:", ica_data.shape)

# ----- Section 6: Real Dataset - Wine Quality (Red) from HuggingFace -----
try:
    # The 'wine_quality' dataset is available on the Hub.
    wine_ds = load_dataset("wine_quality", "red", split="train")
    wine_df = pd.DataFrame(wine_ds)
    # Select all numeric attributes for standardization.
    numeric_cols = wine_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    wine_numeric = wine_df[numeric_cols].copy()
    # Z-score normalization.
    wine_means = wine_numeric.mean()
    wine_stds = wine_numeric.std().replace(0, 1)
    wine_standardized = (wine_numeric - wine_means) / wine_stds
    wine_standardized = wine_standardized.to_numpy()
    # Introduce MCAR missingness: 25% chance per entry is missing.
    p_wine = 0.25
    mask_wine = np.random.rand(*wine_standardized.shape) > p_wine
    wine_masked = wine_standardized.copy().astype(float)
    wine_masked[~mask_wine] = np.nan

    print("\nWine Quality (Red) Dataset:")
    print(" Standardized features shape:", wine_standardized.shape)
    print(" Missingness mask sample (first 5 rows):\n", mask_wine[:5])
    print(" First 5 rows of masked wine data:\n", wine_masked[:5])
except Exception as e:
    print("Error loading Wine Quality dataset:", e)