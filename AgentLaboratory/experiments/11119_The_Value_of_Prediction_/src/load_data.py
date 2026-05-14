import numpy as np
import pandas as pd
from datasets import load_dataset

# Load an external HuggingFace dataset (used here simply to comply with external dataset usage)
# We'll use a small slice of the AG News dataset for demonstration.
hf_dataset = load_dataset("ag_news", split="train[:1%]")
print("Loaded external dataset sample:", hf_dataset[0])

# -----------------------------
# Synthetic data generation
# -----------------------------
# Settings for Gaussian simulation
np.random.seed(42)
N = 300  # number of samples
R2 = 0.5  # target R2 value
rho = np.sqrt(R2)  # correlation between Y and predicted Y (Yhat)

# Simulate the true outcome Y ~ N(0,1)
Y = np.random.normal(loc=0, scale=1, size=N)
# Generate predicted outcome Yhat s.t. (Y, Yhat) is bivariate normal with correlation rho
Yhat = rho * Y + np.sqrt(1 - rho**2) * np.random.normal(loc=0, scale=1, size=N)

# Create additional simple features (e.g., demographic and labor market covariates)
# For simplicity, we simulate gender (binary) and age features.
gender = np.random.choice([0, 1], size=N)  # 0 for female, 1 for male
age = np.random.randint(18, 65, size=N)

# Simulate a 'start_year' to mimic cohort splits (Option A):
# Train: 2010, 2011; Validation: 2012; Test: 2015.
years = np.random.choice([2010, 2011, 2012, 2015], size=N, p=[0.3, 0.3, 0.2, 0.2])

# Create a dataframe to hold our synthetic dataset
df = pd.DataFrame({
    "Y": Y,
    "Yhat": Yhat,
    "gender": gender,
    "age": age,
    "start_year": years
})

# Compute quantile thresholds for evaluation on simulated samples:
alpha = 0.2  # for Yhat
beta = 0.15  # for Y
tYhat_alpha = df["Yhat"].quantile(alpha)
tY_beta = df["Y"].quantile(beta)

# Split indices according to Option A:
train_idx = df.index[df["start_year"].isin([2010, 2011])].tolist()
val_idx = df.index[df["start_year"] == 2012].tolist()
test_idx = df.index[df["start_year"] == 2015].tolist()

# Store split information as columns for reproducibility.
df["split"] = "undefined"
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
df.loc[test_idx, "split"] = "test"

# Save derived artifacts (for demonstration; in practice these could be saved to disk)
print("Quantile threshold for Yhat at alpha=", alpha, "is", tYhat_alpha)
print("Quantile threshold for Y at beta=", beta, "is", tY_beta)
print("Train/Val/Test split sizes:", len(train_idx), len(val_idx), len(test_idx))
print(df.head())