import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate synthetic dataset
class PeerReviewDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = np.random.rand(num_samples, 4)  # Four criteria for reviews
        self.labels = np.random.rand(num_samples)  # RQS values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


# Simple Neural Network
class RQSNet(nn.Module):
    def __init__(self):
        super(RQSNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 32

# Dataset and DataLoader
dataset = PeerReviewDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
model = RQSNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Experiment data storage
experiment_data = {
    "peer_review": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    experiment_data["peer_review"]["losses"]["train"].append(avg_loss)

    print(f"Epoch {epoch + 1}: training_loss = {avg_loss:.4f}")

# Save all metrics at the end
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
