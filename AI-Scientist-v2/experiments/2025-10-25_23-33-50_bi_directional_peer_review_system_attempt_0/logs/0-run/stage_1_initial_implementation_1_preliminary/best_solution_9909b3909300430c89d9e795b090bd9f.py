import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Setting up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate synthetic dataset
class PeerReviewDataset(Dataset):
    def __init__(self, size):
        self.data = np.random.rand(
            size, 2
        )  # Two features: author ratings and review scores
        self.labels = np.random.rand(size)  # RQI labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx], dtype=torch.float32).to(device),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32).to(device),
        }


# Create dataset and dataloaders
dataset = PeerReviewDataset(size=1000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Model, loss function, optimizer
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation loop
epochs = 10
experiment_data = {
    "peer_review": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }
}

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0
    for batch in train_loader:
        features, labels = batch["features"], batch["label"]
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    experiment_data["peer_review"]["losses"]["train"].append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch["features"], batch["label"]
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    experiment_data["peer_review"]["losses"]["val"].append(avg_val_loss)

    # Simulate RQI calculation (for demonstration)
    rqi = 1 - avg_val_loss  # Assume RQI inversely relates to loss
    experiment_data["peer_review"]["metrics"]["train"].append(rqi)

    print(
        f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
