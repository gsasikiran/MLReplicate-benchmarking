import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Synthetic dataset creation
class WorkerDataset(Dataset):
    def __init__(self, num_samples=1000):
        np.random.seed(0)
        self.X = np.random.rand(num_samples, 3)  # 3 features
        self.y = (
            self.X[:, 0] * 0.5
            + self.X[:, 1] * -0.3
            + self.X[:, 2] * 0.2
            + np.random.normal(0, 0.1, num_samples)
        ) * 100  # WIS target
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Neural network model
class WISModel(nn.Module):
    def __init__(self):
        super(WISModel, self).__init__()
        self.fc = nn.Sequential(nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, x):
        return self.fc(x)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize dataset and dataloaders
dataset = WorkerDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = WISModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data storage
experiment_data = {
    "worker_dataset": {
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
    train_loss = 0
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    experiment_data["worker_dataset"]["losses"]["train"].append(train_loss)

    # Print training loss
    print(f"Epoch {epoch + 1}: training_loss = {train_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
