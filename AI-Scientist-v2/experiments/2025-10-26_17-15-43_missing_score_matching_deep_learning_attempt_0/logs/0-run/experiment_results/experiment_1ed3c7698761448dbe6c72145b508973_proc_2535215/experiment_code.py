import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Create a working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_samples, num_features = 1000, 20
np.random.seed(42)


# Function to create synthetic data with missing values
def create_synthetic_data(missing_rate):
    X = np.random.rand(num_samples, num_features)
    missing_mask = np.random.rand(*X.shape) < missing_rate
    X_missing = np.where(missing_mask, np.nan, X)
    return X_missing, X  # Return both missing data and original


# Simple dataset class
class SyntheticDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)


# Neural network model
class SimpleNN(nn.Module):
    def __init__(self, hidden_layer_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Configuration
hidden_layer_sizes = [32, 64, 128]
missing_rates = [0.0, 0.1, 0.3, 0.5]
experiment_data = {"multiple_synthetic_datasets": {}}


# Impute missing values using mean imputation
def impute_data(X):
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X


# Iterate over different missing rates
for missing_rate in missing_rates:
    dataset_name = f"missing_rate_{int(missing_rate * 100)}"
    X_imputed, X_original = create_synthetic_data(missing_rate)
    X_imputed = impute_data(X_imputed)
    X_train, X_val = train_test_split(X_imputed, test_size=0.2, random_state=42)

    train_dataset = SyntheticDataset(X_train)
    val_dataset = SyntheticDataset(X_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    experiment_data["multiple_synthetic_datasets"][dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "mdie": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for hidden_layer_size in hidden_layer_sizes:
        model = SimpleNN(hidden_layer_size).to(device)
        criterion = nn.MSELoss(reduction="sum")
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            total_mdie = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)

                mask = ~torch.isnan(batch)
                loss = (
                    criterion(output[mask], batch[mask])
                    if mask.any()
                    else torch.tensor(0.0, device=device)
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total_mdie += (
                    torch.mean(torch.abs(output[mask] - batch[mask])).item()
                    if mask.any()
                    else 0
                )

            train_loss /= len(train_loader)
            total_mdie /= len(train_loader) if len(train_loader) > 0 else 1

            model.eval()
            val_loss = 0.0
            total_val_mdie = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)

                    mask = ~torch.isnan(batch)
                    val_loss += (
                        criterion(output[mask], batch[mask])
                        if mask.any()
                        else torch.tensor(0.0, device=device)
                    )
                    total_val_mdie += (
                        torch.mean(torch.abs(output[mask] - batch[mask])).item()
                        if mask.any()
                        else 0
                    )
            val_loss /= len(val_loader)
            total_val_mdie /= len(val_loader) if len(val_loader) > 0 else 1

            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "train"
            ].append(train_loss)
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "val"
            ].append(val_loss)
            experiment_data["multiple_synthetic_datasets"][dataset_name]["mdie"][
                "train"
            ].append(total_mdie)
            experiment_data["multiple_synthetic_datasets"][dataset_name]["mdie"][
                "val"
            ].append(total_val_mdie)

            print(
                f"Dataset: {dataset_name}, Hidden Layer Size {hidden_layer_size}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, MDIE (train) = {total_mdie:.4f}, MDIE (val) = {total_val_mdie:.4f}"
            )

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
