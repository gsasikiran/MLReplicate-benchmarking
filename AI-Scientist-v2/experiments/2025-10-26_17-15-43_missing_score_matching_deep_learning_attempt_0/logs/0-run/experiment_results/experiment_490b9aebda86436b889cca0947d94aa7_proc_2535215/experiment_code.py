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


# Impute missing values with the mean for model training
X_imputed = np.where(np.isnan(X_missing), np.nanmean(X_missing, axis=0), X_missing)
X_train, X_val = train_test_split(X_imputed, test_size=0.2, random_state=42)

train_dataset = SyntheticDataset(X_train)
val_dataset = SyntheticDataset(X_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Modify SimpleNN to include Batch Normalization
class SimpleNN(nn.Module):
    def __init__(self, hidden_layer_size, use_batch_norm=False):
        super(SimpleNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(num_features, hidden_layer_size)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        if self.use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if self.use_batch_norm:
            x = self.bn1(x)
        x = torch.relu(self.fc2(x))
        if self.use_batch_norm:
            x = self.bn2(x)
        return self.fc3(x)


# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuring ablation study for Batch Normalization
hidden_layer_sizes = [32, 64, 128]
experiment_data = {
    "impact_of_batch_normalization": {
        "with_bn": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
        },
        "without_bn": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
        },
    }
}

num_epochs = 50

# Experiment with Batch Normalization
for hidden_layer_size in hidden_layer_sizes:
    # Experiment with Batch Normalization
    model_with_bn = SimpleNN(hidden_layer_size, use_batch_norm=True).to(device)
    optimizer = optim.Adam(model_with_bn.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model_with_bn.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model_with_bn(batch)
            loss = nn.MSELoss()(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model_with_bn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model_with_bn(batch)
                loss = nn.MSELoss()(output, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        experiment_data["impact_of_batch_normalization"]["with_bn"]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["impact_of_batch_normalization"]["with_bn"]["losses"][
            "val"
        ].append(val_loss)
        experiment_data["impact_of_batch_normalization"]["with_bn"]["metrics"][
            "train"
        ].append(train_loss)
        experiment_data["impact_of_batch_normalization"]["with_bn"]["metrics"][
            "val"
        ].append(val_loss)

        print(
            f"With BN - Hidden Layer Size {hidden_layer_size}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
        )

    # Experiment without Batch Normalization
    model_without_bn = SimpleNN(hidden_layer_size, use_batch_norm=False).to(device)
    optimizer = optim.Adam(model_without_bn.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model_without_bn.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model_without_bn(batch)
            loss = nn.MSELoss()(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model_without_bn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model_without_bn(batch)
                loss = nn.MSELoss()(output, batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        experiment_data["impact_of_batch_normalization"]["without_bn"]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["impact_of_batch_normalization"]["without_bn"]["losses"][
            "val"
        ].append(val_loss)
        experiment_data["impact_of_batch_normalization"]["without_bn"]["metrics"][
            "train"
        ].append(train_loss)
        experiment_data["impact_of_batch_normalization"]["without_bn"]["metrics"][
            "val"
        ].append(val_loss)

        print(
            f"Without BN - Hidden Layer Size {hidden_layer_size}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
