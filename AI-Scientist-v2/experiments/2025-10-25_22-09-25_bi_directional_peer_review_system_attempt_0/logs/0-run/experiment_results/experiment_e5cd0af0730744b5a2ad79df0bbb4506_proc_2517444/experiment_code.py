# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import subprocess
import sys

# Ensure sklearn is installed
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Handle GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic dataset
np.random.seed(0)
num_samples = 1000
features = np.random.rand(
    num_samples, 3
)  # e.g., thoroughness, constructiveness, acceptance rate
labels = np.random.rand(num_samples)  # RQS values

# Prepare DataLoader
X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_scaled_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
dataset_scaled = TensorDataset(X_scaled_tensor, y_tensor)
train_loader_scaled = DataLoader(dataset_scaled, batch_size=32, shuffle=True)


# Define simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Prepare data storage
experiment_data = {
    "Input_Feature_Scaling_Impact": {
        "unscaled_data": {
            "metrics": {"train": [], "validation": []},
            "losses": {"train": [], "validation": []},
            "predictions": [],
            "ground_truth": [],
        },
        "scaled_data": {
            "metrics": {"train": [], "validation": []},
            "losses": {"train": [], "validation": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Hyperparameter tuning for number of hidden units
hidden_units_list = [16, 32, 48, 64]
num_epochs = 10

for hidden_units in hidden_units_list:
    print(f"\nTraining with hidden units: {hidden_units}")

    # Train with unscaled data
    model = SimpleNN(hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in zip(("features", "labels"), batch)}
            optimizer.zero_grad()
            outputs = model(batch["features"])
            loss = criterion(outputs.squeeze(), batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        rqi = 1 / (1 + avg_loss)  # Placeholder for actual RQI calculation
        experiment_data["Input_Feature_Scaling_Impact"]["unscaled_data"]["losses"][
            "train"
        ].append(avg_loss)
        experiment_data["Input_Feature_Scaling_Impact"]["unscaled_data"]["metrics"][
            "train"
        ].append(rqi)
        print(f"Unscaled Epoch {epoch+1}: loss = {avg_loss:.4f}, RQI = {rqi:.4f}")

    # Train with scaled data
    model = SimpleNN(hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader_scaled:
            batch = {k: v.to(device) for k, v in zip(("features", "labels"), batch)}
            optimizer.zero_grad()
            outputs = model(batch["features"])
            loss = criterion(outputs.squeeze(), batch["labels"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader_scaled)
        rqi = 1 / (1 + avg_loss)  # Placeholder for actual RQI calculation
        experiment_data["Input_Feature_Scaling_Impact"]["scaled_data"]["losses"][
            "train"
        ].append(avg_loss)
        experiment_data["Input_Feature_Scaling_Impact"]["scaled_data"]["metrics"][
            "train"
        ].append(rqi)
        print(f"Scaled Epoch {epoch+1}: loss = {avg_loss:.4f}, RQI = {rqi:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
