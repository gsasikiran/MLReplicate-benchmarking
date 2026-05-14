import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic dataset creation
np.random.seed(0)
num_samples = 1000
X = np.random.rand(
    num_samples, 3
)  # Features: job displacement, wage fluctuation, worker satisfaction
y = (X[:, 0] * 0.5 + X[:, 1] * -0.3 + X[:, 2] * 0.2).reshape(-1, 1)  # Hypothetical WIS

# Split into training and validation sets
split = int(0.8 * num_samples)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Dataloader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleNN().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data storage
experiment_data = {
    "simple_nn_experiment": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            val_loss += criterion(outputs, y_batch).item()

    val_loss /= len(val_loader)
    val_metric = -val_loss  # Assume inversion for maximization

    # Save metrics
    experiment_data["simple_nn_experiment"]["losses"]["train"].append(loss.item())
    experiment_data["simple_nn_experiment"]["losses"]["val"].append(val_loss)
    experiment_data["simple_nn_experiment"]["metrics"]["train"].append(-loss.item())
    experiment_data["simple_nn_experiment"]["metrics"]["val"].append(val_metric)

    print(f"Epoch {epoch+1}: validation_loss = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
