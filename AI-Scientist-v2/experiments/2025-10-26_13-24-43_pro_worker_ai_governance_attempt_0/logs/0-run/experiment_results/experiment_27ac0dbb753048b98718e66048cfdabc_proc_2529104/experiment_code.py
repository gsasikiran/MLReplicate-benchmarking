import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic dataset
n_samples = 1000
X = np.random.rand(n_samples, 3)  # Features: AI intervention metrics
y = np.random.rand(n_samples)  # Label: Pro-Worker Impact Score (PWIS)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model, loss function and optimizer
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare experiment data dictionary
experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in zip(["X", "y"], batch)}
        optimizer.zero_grad()
        outputs = model(batch["X"]).squeeze()
        loss = criterion(outputs, batch["y"])
        loss.backward()
        optimizer.step()

    # Store loss
    experiment_data["synthetic_data"]["losses"]["train"].append(loss.item())
    print(f"Epoch {epoch + 1}/{num_epochs}: train_loss = {loss.item():.4f}")

    # Evaluate the PWIS (simulated)
    train_metric = np.random.rand()  # Replace with actual metric calculation
    experiment_data["synthetic_data"]["metrics"]["train"].append(train_metric)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
