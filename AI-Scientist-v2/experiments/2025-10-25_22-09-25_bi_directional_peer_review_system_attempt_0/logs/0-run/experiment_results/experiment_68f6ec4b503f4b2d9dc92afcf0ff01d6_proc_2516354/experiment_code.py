import os
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


# Define simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Hyperparameter tuning setup for weight decay
weight_decay_values = [0.0, 0.01, 0.1, 0.5]
experiment_data = {
    "weight_decay_tuning": {
        "synthetic_data": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Training loop for different weight decay values
num_epochs = 10
for weight_decay in weight_decay_values:
    print(f"\nTraining with weight decay: {weight_decay}")
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

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
        experiment_data["weight_decay_tuning"]["synthetic_data"]["losses"][
            "train"
        ].append(avg_loss)

        # Calculate RQS (for simplicity using average here)
        rqs = 1 - avg_loss  # Placeholder for actual RQS
        experiment_data["weight_decay_tuning"]["synthetic_data"]["metrics"][
            "train"
        ].append(rqs)

        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
