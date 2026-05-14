# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Generate synthetic data
np.random.seed(42)
num_samples = 1000
job_displacement = np.random.rand(num_samples)  # 0 to 1
wage_change = np.random.rand(num_samples)  # 0 to 1
worker_satisfaction = np.random.rand(num_samples)  # 0 to 1
WIS = (
    job_displacement * 0.3 + wage_change * 0.4 + worker_satisfaction * 0.3
)  # Weighted sum to create WIS

# Train/test split
X = np.vstack((job_displacement, wage_change, worker_satisfaction)).T
y = WIS
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Initialize model, loss, and optimizer
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track metrics
experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in zip(("inputs", "targets"), batch)}
        optimizer.zero_grad()
        outputs = model(batch["inputs"])
        loss = criterion(outputs, batch["targets"].view(-1, 1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    experiment_data["synthetic_data"]["losses"]["train"].append(avg_train_loss)

    # Validation step
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in zip(("inputs", "targets"), batch)}
            outputs = model(batch["inputs"])
            loss = criterion(outputs, batch["targets"].view(-1, 1))
            total_val_loss += loss.item()
            experiment_data["synthetic_data"]["predictions"].extend(
                outputs.cpu().numpy()
            )
            experiment_data["synthetic_data"]["ground_truth"].extend(
                batch["targets"].cpu().numpy()
            )

    avg_val_loss = total_val_loss / len(val_loader)
    experiment_data["synthetic_data"]["losses"]["val"].append(avg_val_loss)
    print(f"Epoch {epoch+1}: validation_loss = {avg_val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
