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


# Define a simple feedforward neural network with dropout
class SimpleNN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_features)  # Output same as input features

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Hyperparameter tuning for dropout_rate
dropout_rates = [0.0, 0.2, 0.4, 0.6]
experiment_data = {
    "dropout_tuning": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 50
for dropout_rate in dropout_rates:
    model = SimpleNN(dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)  # Training with the same input as output
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
        experiment_data["dropout_tuning"]["synthetic_dataset"]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["dropout_tuning"]["synthetic_dataset"]["losses"]["val"].append(
            val_loss
        )
        experiment_data["dropout_tuning"]["synthetic_dataset"]["metrics"][
            "train"
        ].append(train_loss)
        experiment_data["dropout_tuning"]["synthetic_dataset"]["metrics"]["val"].append(
            val_loss
        )

        print(
            f"Dropout Rate: {dropout_rate}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
