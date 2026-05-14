import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
X_train_imp, X_val_imp = train_test_split(X_imputed, test_size=0.2, random_state=42)

# Normalize the imputed data
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_imp)
X_val_norm = scaler.transform(X_val_imp)

# Create datasets
train_dataset_imp = SyntheticDataset(X_train_imp)
val_dataset_imp = SyntheticDataset(X_val_imp)
train_dataset_norm = SyntheticDataset(X_train_norm)
val_dataset_norm = SyntheticDataset(X_val_norm)

# DataLoaders
train_loader_imp = DataLoader(train_dataset_imp, batch_size=32, shuffle=True)
val_loader_imp = DataLoader(val_dataset_imp, batch_size=32, shuffle=False)
train_loader_norm = DataLoader(train_dataset_norm, batch_size=32, shuffle=True)
val_loader_norm = DataLoader(val_dataset_norm, batch_size=32, shuffle=False)


# Neural network definition
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
experiment_data = {
    "input_data_normalization": {
        "mean_imputation": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "data_normalization": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# Function to train and validate the model
def train_and_validate(hidden_layer_size, train_loader, val_loader, experiment_key):
    model = SimpleNN(hidden_layer_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)  # Overwriting input as output
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
        experiment_data["input_data_normalization"][experiment_key]["losses"][
            "train"
        ].append(train_loss)
        experiment_data["input_data_normalization"][experiment_key]["losses"][
            "val"
        ].append(val_loss)
        experiment_data["input_data_normalization"][experiment_key]["metrics"][
            "train"
        ].append(train_loss)
        experiment_data["input_data_normalization"][experiment_key]["metrics"][
            "val"
        ].append(val_loss)

        print(
            f"{experiment_key}, Hidden Layer Size {hidden_layer_size}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}"
        )


# Train and validate on mean imputation
for hidden_layer_size in hidden_layer_sizes:
    train_and_validate(
        hidden_layer_size, train_loader_imp, val_loader_imp, "mean_imputation"
    )

# Train and validate on normalized data
for hidden_layer_size in hidden_layer_sizes:
    train_and_validate(
        hidden_layer_size, train_loader_norm, val_loader_norm, "data_normalization"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
