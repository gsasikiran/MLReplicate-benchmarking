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
features = np.random.rand(num_samples, 3)
labels = np.random.rand(num_samples)

# Prepare DataLoader
X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define simple feedforward neural network with Batch Normalization
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        return self.fc2(x)


# Hyperparameter tuning configuration
learning_rates = [0.001, 0.01, 0.1]
experiment_data = {}

# Training loop for different learning rates
for lr in learning_rates:
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    experiment_data[f"hyperparam_tuning_lr_{lr}"] = {
        "synthetic_data": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }

    num_epochs = 10
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
        experiment_data[f"hyperparam_tuning_lr_{lr}"]["synthetic_data"]["losses"][
            "train"
        ].append(avg_loss)

        # Calculate RQS
        rqs = 1 - avg_loss
        experiment_data[f"hyperparam_tuning_lr_{lr}"]["synthetic_data"]["metrics"][
            "train"
        ].append(rqs)

        print(f"LR: {lr}, Epoch {epoch+1}: loss = {avg_loss:.4f}, RQS = {rqs:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
