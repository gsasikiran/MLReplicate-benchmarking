import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer
from torch.utils.data import DataLoader, Dataset

# Create a working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Generate synthetic data with missing values
np.random.seed(42)
num_samples, num_features = 1000, 20
X = np.random.rand(num_samples, num_features)
missing_mask = np.random.rand(*X.shape) < 0.2  # 20% missingness
X_missing = np.where(missing_mask, np.nan, X)


# Simple dataset class
class SyntheticDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)


# Simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, hidden_layer_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, num_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Function to apply imputation methods
def impute_data(X, method):
    if method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "median":
        imputer = SimpleImputer(strategy="median")
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "iterative":
        imputer = IterativeImputer()
    else:
        raise ValueError("Unknown imputation method")

    return imputer.fit_transform(X)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "imputation_methods": {
        "mean": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "mdie": [],
        },
        "median": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "mdie": [],
        },
        "knn": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "mdie": [],
        },
        "iterative": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "mdie": [],
        },
    }
}

hidden_layer_sizes = [32, 64, 128]

for imputation_method in experiment_data["imputation_methods"]:
    X_imputed = impute_data(X_missing, imputation_method)
    X_train, X_val = train_test_split(X_imputed, test_size=0.2, random_state=42)

    train_dataset = SyntheticDataset(X_train)
    val_dataset = SyntheticDataset(X_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for hidden_layer_size in hidden_layer_sizes:
        model = SimpleNN(hidden_layer_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training and validation loop
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            total_mdie = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    loss = criterion(output, batch)
                    val_loss += loss.item()
                    total_mdie += torch.mean(torch.abs(output - batch)).item()
            val_loss /= len(val_loader)
            total_mdie /= len(val_loader)

            # Store metrics
            experiment_data["imputation_methods"][imputation_method]["losses"][
                "train"
            ].append(train_loss)
            experiment_data["imputation_methods"][imputation_method]["losses"][
                "val"
            ].append(val_loss)
            experiment_data["imputation_methods"][imputation_method]["mdie"].append(
                total_mdie
            )

            print(
                f"Imputation {imputation_method}, Hidden Layer Size {hidden_layer_size}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, MDIE = {total_mdie:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
