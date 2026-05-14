# Set random seed
import random
import numpy as np
import torch

seed = 1
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
from sklearn.model_selection import train_test_split

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic data generation
np.random.seed(42)
num_samples = 1000
worker_attributes = np.random.rand(
    num_samples, 3
)  # Example attributes: workload, job security, retraining opportunities
satisfaction = (
    0.5 * worker_attributes[:, 0]
    + 0.3 * worker_attributes[:, 1]
    + 0.2 * worker_attributes[:, 2]
    + np.random.normal(0, 0.05, num_samples)
)

X_train, X_val, y_train, y_val = train_test_split(
    worker_attributes, satisfaction, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data structure
experiment_data = {
    "synthetic_dataset": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs.squeeze(), y_train_tensor.to(device))
    loss.backward()
    optimizer.step()

    # Track training loss
    experiment_data["synthetic_dataset"]["losses"]["train"].append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor.to(device))
        val_loss = criterion(val_outputs.squeeze(), y_val_tensor.to(device))

    # Track validation loss
    experiment_data["synthetic_dataset"]["losses"]["val"].append(val_loss.item())
    print(f"Epoch {epoch + 1}: validation_loss = {val_loss:.4f}")

    # Calculate PWIS (using dummy transformation for simplicity)
    PWIS = 1 - val_loss.item()  # Higher score is better
    experiment_data["synthetic_dataset"]["metrics"]["train"].append(PWIS)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Optional: Save predictions and ground truth for analysis
experiment_data["synthetic_dataset"]["predictions"] = val_outputs.cpu().numpy()
experiment_data["synthetic_dataset"]["ground_truth"] = y_val_tensor.cpu().numpy()

# Output PWIS score
print(f"Final Pro-Worker Impact Score (PWIS): {PWIS:.4f}")
