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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(0)
num_samples = 1000
X = np.random.rand(num_samples, 3)  # Simulated features: clarity, depth, relevance
RQS = np.clip(
    X[:, 0] * 0.5
    + X[:, 1] * 0.3
    + X[:, 2] * 0.2
    + np.random.normal(0, 0.05, num_samples),
    0,
    1,
)

X_train, X_val, y_train, y_val = train_test_split(
    X, RQS, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)


# Define the neural network model
class RQSModel(nn.Module):
    def __init__(self):
        super(RQSModel, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Hyperparameter tuning: batch sizes
batch_sizes = [8, 16, 32, 64]
num_epochs = 50

experiment_data = {
    "hyperparam_tuning_batch_size": {
        "RQS": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for batch_size in batch_sizes:
    model = RQSModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i : i + batch_size]
            y_batch = y_train_tensor[i : i + batch_size]
            optimizer.zero_grad()

            # Forward pass
            y_train_pred = model(X_batch)
            train_loss = criterion(y_train_pred.squeeze(), y_batch)
            train_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = criterion(y_val_pred.squeeze(), y_val_tensor)

        # Update metrics
        experiment_data["hyperparam_tuning_batch_size"]["RQS"]["metrics"][
            "train"
        ].append(
            1 - train_loss.item()
        )  # RQS
        experiment_data["hyperparam_tuning_batch_size"]["RQS"]["losses"][
            "train"
        ].append(train_loss.item())
        experiment_data["hyperparam_tuning_batch_size"]["RQS"]["metrics"]["val"].append(
            1 - val_loss.item()
        )  # RQS
        experiment_data["hyperparam_tuning_batch_size"]["RQS"]["losses"]["val"].append(
            val_loss.item()
        )

        print(
            f"Batch size: {batch_size}, Epoch {epoch + 1}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}"
        )

    experiment_data["hyperparam_tuning_batch_size"]["RQS"]["predictions"].append(
        y_val_pred.cpu().numpy()
    )
    experiment_data["hyperparam_tuning_batch_size"]["RQS"]["ground_truth"].append(
        y_val_tensor.cpu().numpy()
    )

# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
