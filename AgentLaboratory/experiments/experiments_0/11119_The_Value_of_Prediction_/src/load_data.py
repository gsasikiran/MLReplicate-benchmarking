import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# Load an external dataset from HuggingFace (we use ag_news)
external_ds = load_dataset("ag_news", split="train")
df_external = pd.DataFrame(external_ds)
n_external = len(df_external)

# Use the size of the external dataset for our synthetic simulations
np.random.seed(123)
n = n_external

# ----- Gaussian Track Simulation -----
R2 = 0.7                       # target R^2 value for Gaussian simulation
rho = np.sqrt(R2)              # induced correlation
sigma = 1.0                    # standard deviation for Y
# Simulate outcome Y ~ N(0,1)
Y_gauss = np.random.normal(0, sigma, n)
# Simulate predictor Yhat with desired correlation: Yhat = rho * Y + sqrt(1 - rho^2)*epsilon
epsilon_gauss = np.random.normal(0, 1, n)
Yhat_gauss = rho * Y_gauss + np.sqrt(1 - rho**2) * epsilon_gauss
alpha = 0.75                   # quantile level for Gaussian track
tYhat_gauss = np.quantile(Yhat_gauss, alpha)
tY_gauss = np.quantile(Y_gauss, alpha)

# ----- Log-Normal Track Simulation -----
# Here, assume log Y ~ N(0,1); then Y = exp(logY) up to multiplicative noise
log_Y = np.random.normal(0, sigma, n)
Y_ln = np.exp(log_Y)
# Create correlated log-transformed predictor: log_Yhat with same desired correlation
epsilon_ln = np.random.normal(0, 1, n)
log_Yhat = rho * log_Y + np.sqrt(1 - rho**2) * epsilon_ln
Yhat_ln = np.exp(log_Yhat)
# Introduce multiplicative noise factor: u ~ lognormal(0, gamma^2)
gamma = 0.3
mult_noise = np.exp(np.random.normal(0, gamma, n))
Y_ln = Y_ln * mult_noise
beta = 0.2                    # quantile level for lognormal track (lower tail)
tYhat_ln = np.quantile(Yhat_ln, beta)
tY_ln = np.quantile(Y_ln, beta)

# ----- Combine Simulation Tracks with External Data Details -----
# Create a DataFrame that combines both simulation tracks.
combined_df = pd.DataFrame({
    "id": list(range(2 * n)),
    "Y": np.concatenate([Y_gauss, Y_ln]),
    "Yhat": np.concatenate([Yhat_gauss, Yhat_ln]),
    "track": ["gaussian"] * n + ["lognormal"] * n
})
# Incorporate an external feature from the ag_news dataset (for example, the 'label' column)
# Duplicate the external label for both simulation tracks.
if "label" in df_external.columns:
    combined_df["external_label"] = list(df_external["label"]) * 2
else:
    combined_df["external_label"] = [None] * (2 * n)

# Shuffle the combined dataset.
shuffled_indices = np.random.permutation(len(combined_df))
combined_df = combined_df.iloc[shuffled_indices].reset_index(drop=True)

# ----- Create Dataset Splits -----
total_samples = len(combined_df)
n_train = int(0.65 * total_samples)
n_val = int(0.20 * total_samples)
n_test = total_samples - n_train - n_val

train_df = combined_df.iloc[:n_train]
val_df = combined_df.iloc[n_train:n_train + n_val]
test_df = combined_df.iloc[n_train + n_val:]

train_dataset = Dataset.from_dict(train_df.to_dict(orient="list"))
val_dataset = Dataset.from_dict(val_df.to_dict(orient="list"))
test_dataset = Dataset.from_dict(test_df.to_dict(orient="list"))

data_splits = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# ----- Print Diagnostics -----
print("Diagnostics:")
print("External dataset size used for simulation:", n)
print("Gaussian track (alpha = 75%) quantiles -> Yhat:", tYhat_gauss, " Y:", tY_gauss)
print("Lognormal track (beta = 20%) quantiles -> Yhat:", tYhat_ln, " Y:", tY_ln)
print("Final dataset splits:")
print("  Train samples:", len(data_splits["train"]))
print("  Validation samples:", len(data_splits["validation"]))
print("  Test samples:", len(data_splits["test"]))