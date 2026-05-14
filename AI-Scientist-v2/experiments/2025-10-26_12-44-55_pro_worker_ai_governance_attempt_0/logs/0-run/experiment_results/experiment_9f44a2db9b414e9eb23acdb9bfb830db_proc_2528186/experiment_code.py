import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic dataset
np.random.seed(42)
data_size = 1000
job_displacement = np.random.uniform(0, 1, data_size)  # Job displacement rates
wage_change = np.random.uniform(-1, 1, data_size)  # Wage fluctuations
retraining_access = np.random.uniform(0, 1, data_size)  # Access to retraining programs
WIS = (
    0.3 * job_displacement + 0.5 * (1 - wage_change) + 0.2 * retraining_access
)  # WIS calculation

X = np.column_stack((job_displacement, wage_change, retraining_access))
y = WIS

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Dataset class
class WorkerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Prepare experiment data storage
experiment_data = {
    "unscaled": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "standardized": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "normalized": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# List of learning rates to test
learning_rates = [0.001, 0.01, 0.0001]
num_epochs = 50


# Training function
def train_model(X_train, y_train, X_val, y_val, lr, scaling_type):
    train_dataset = WorkerDataset(X_train, y_train)
    val_dataset = WorkerDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data[scaling_type]["losses"]["train"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
                experiment_data[scaling_type]["predictions"].extend(
                    outputs.squeeze().cpu().numpy()
                )
                experiment_data[scaling_type]["ground_truth"].extend(
                    targets.cpu().numpy()
                )

        val_loss /= len(val_loader)
        experiment_data[scaling_type]["losses"]["val"].append(val_loss)
        PWIS = 1 - val_loss  # Compute Pro-Worker Impact Score
        experiment_data[scaling_type]["metrics"]["val"].append(PWIS)

        print(f"Epoch {epoch + 1}: validation_loss = {val_loss:.4f}, PWIS = {PWIS:.4f}")


# Train without scaling
train_model(X_train, y_train, X_val, y_val, 0.001, "unscaled")

# Standardize features
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_val_standard = scaler_standard.transform(X_val)

# Train with standardization
train_model(X_train_standard, y_train, X_val_standard, y_val, 0.001, "standardized")

# Normalize features
scaler_normal = MinMaxScaler()
X_train_normalized = scaler_normal.fit_transform(X_train)
X_val_normalized = scaler_normal.transform(X_val)

# Train with normalization
train_model(X_train_normalized, y_train, X_val_normalized, y_val, 0.001, "normalized")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
