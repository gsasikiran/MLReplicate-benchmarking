import numpy as np
import pandas as pd
from datasets import load_dataset

np.random.seed(0)

# -----------------------------
# Load external HuggingFace dataset: CIFAR-10
# -----------------------------
# Load the training split and convert images to float32 arrays (flattened)
cifar = load_dataset("cifar10", split="train")
# Use the first 50 examples for a quick demo
cifar_imgs = [np.array(item["img"]).astype(np.float32).flatten() for item in cifar.select(range(50))]
cifar_data = np.stack(cifar_imgs)
print("CIFAR-10 flattened data shape:", cifar_data.shape)
# Apply MCAR missingness with probability 0.3
p_missing = 0.3
mask_cifar = np.random.rand(*cifar_data.shape) > p_missing
cifar_data_masked = cifar_data.copy()
cifar_data_masked[~mask_cifar] = np.nan
print("CIFAR-10 masked data sample (first row):", cifar_data_masked[0])

# -----------------------------
# Generate Synthetic 10-D Gaussian Data with dependency and truncation
# -----------------------------
n_samples = 400
d = 10
eig_vals = np.random.uniform(0.5, 1.5, d)
Q, _ = np.linalg.qr(np.random.randn(d, d))
cov_matrix = Q @ np.diag(eig_vals) @ Q.T
mean_vector = np.full(d, 0.5)
gaussian_data = np.random.multivariate_normal(mean_vector, cov_matrix, size=n_samples)
# Impose dependency: Last variable = 0.5 * first variable + noise
noise = np.random.normal(loc=gaussian_data[:, 0], scale=np.std(gaussian_data[:, 0]), size=n_samples)
gaussian_data[:, -1] = 0.5 * gaussian_data[:, 0] + 0.5 * noise
# Truncate first three dimensions at their 10th percentiles (simulate boundary effects)
for i in range(3):
    thr = np.percentile(gaussian_data[:, i], 10)
    gaussian_data[:, i] = np.minimum(gaussian_data[:, i], thr)
    print(f"Gaussian dim {i+1} truncated at 10th percentile: {thr:.4f}")
# Apply MCAR missingness with probability 0.3
mask_gauss = np.random.rand(*gaussian_data.shape) > p_missing
gaussian_data_masked = gaussian_data.copy()
gaussian_data_masked[~mask_gauss] = np.nan
print("Gaussian masked data sample (first row):", gaussian_data_masked[0])
df_gaussian = pd.DataFrame(gaussian_data_masked, columns=[f"G{i+1}" for i in range(d)])
df_gaussian["obs_count"] = np.sum(~np.isnan(gaussian_data_masked), axis=1)
print("Synthetic Gaussian DataFrame sample:")
print(df_gaussian.head())

# -----------------------------
# Generate Synthetic ICA-like Data using Laplace distribution and nonlinearity
# -----------------------------
n_ica = 300
d_ica = 10
# Generate Laplace-distributed samples for heavy-tailed behavior
laplace_samples = np.random.laplace(loc=0.0, scale=1.0, size=(n_ica, d_ica))
# Introduce nonlinearity: simulate interactions by applying a tanh and squaring operation
ica_data = np.tanh(laplace_samples) * (laplace_samples ** 2)
print("ICA-like data generated with shape:", ica_data.shape)
# Apply MCAR missingness with probability 0.3
mask_ica = np.random.rand(*ica_data.shape) > p_missing
ica_data_masked = ica_data.copy()
ica_data_masked[~mask_ica] = np.nan
print("ICA-like masked data sample (first row):", ica_data_masked[0])
df_ica = pd.DataFrame(ica_data_masked, columns=[f"ICA{i+1}" for i in range(d_ica)])
df_ica["obs_count"] = np.sum(~np.isnan(ica_data_masked), axis=1)
print("Synthetic ICA-like DataFrame sample:")
print(df_ica.head())

# -----------------------------
# Generate Synthetic Gaussian Graphical Model (GGM) Data using a star graph structure
# -----------------------------
n_ggm = 250
d_ggm = 10
# Construct a precision matrix for a star graph (first feature is the hub)
precision = np.eye(d_ggm)
for j in range(1, d_ggm):
    precision[0, j] = -0.3
    precision[j, 0] = -0.3
    precision[j, j] = 1.0
cov_ggm = np.linalg.inv(precision)
mean_ggm = np.zeros(d_ggm)
ggm_data = np.random.multivariate_normal(mean_ggm, cov_ggm, size=n_ggm)
print("GGM data generated with shape:", ggm_data.shape)
# Apply MCAR missingness with probability 0.3 on GGM data
mask_ggm = np.random.rand(*ggm_data.shape) > p_missing
ggm_data_masked = ggm_data.copy()
ggm_data_masked[~mask_ggm] = np.nan
print("GGM masked data sample (first row):", ggm_data_masked[0])
df_ggm = pd.DataFrame(ggm_data_masked, columns=[f"GGM{i+1}" for i in range(d_ggm)])
df_ggm["obs_count"] = np.sum(~np.isnan(ggm_data_masked), axis=1)
print("Synthetic GGM DataFrame sample:")
print(df_ggm.head())

print("Data preparation complete.")