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


# Function to generate synthetic datasets with different missingness patterns
def generate_missing_data(X, pattern="mcar"):
    np.random.seed(42)
    if pattern == "mcar":  # Missing Completely at Random
        missing_mask = np.random.rand(*X.shape) < 0.2
        X_missing = np.where(missing_mask, np.nan, X)

    elif pattern == "mar":  # Missing at Random
        missing_mask = np.random.rand(*X.shape) < (
            0.2 * (1 - X.mean(axis=1, keepdims=True))
        )
        X_missing = np.where(missing_mask, np.nan, X)

    elif pattern == "nmar":  # Not Missing at Random
        missing_mask = (X > 0.5) & (np.random.rand(*X.shape) < 0.2)
        X_missing = np.where(missing_mask, np.nan, X)

    return X_missing


# Generate synthetic data
np.random.seed(42)
num_samples, num_features = 1000, 20
X = np.random.rand(num_samples, num_features)

# Define ablation study data structure
experiment_data = {
    "missing_data_patterns": {
        "mcar": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "mar": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "nmar": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# Simple dataset class
class SyntheticDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)


# Modify the neural network to accept hidden_layer_size as a parameter
class SimpleNN(nn.Module):
    def __init__(self, hidden_layer_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(
            hidden_layer_size, num_features
        )  # Output same as input features

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuring hyperparameter tuning for hidden layer sizes
hidden_layer_sizes = [32, 64, 128]
num_epochs = 50

# Train on each pattern dataset
for pattern in ["mcar", "mar", "nmar"]:
    X_missing = generate_missing_data(X, pattern)
    X_imputed = np.where(np.isnan(X_missing), np.nanmean(X_missing, axis=0), X_missing)
    X_train, X_val = train_test_split(X_imputed, test_size=0.2, random_state=42)

    train_dataset = SyntheticDataset(X_train)
    val_dataset = SyntheticDataset(X_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    for hidden_layer_size in hidden_layer_sizes:
        model = SimpleNN(hidden_layer_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Store experiment metrics for each hidden layer size
        experiment_data["missing_data_patterns"][pattern]["losses"]["train"] = []
        experiment_data["missing_data_patterns"][pattern]["losses"]["val"] = []

        # Training and validation loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(
                    output, batch
                )  # Training with the same input as output
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    loss = criterion(output, batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Store metrics
            experiment_data["missing_data_patterns"][pattern]["losses"]["train"].append(
                train_loss
            )
            experiment_data["missing_data_patterns"][pattern]["losses"]["val"].append(
                val_loss
            )

            print(
                f"Pattern {pattern}, Hidden Layer Size {hidden_layer_size}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
