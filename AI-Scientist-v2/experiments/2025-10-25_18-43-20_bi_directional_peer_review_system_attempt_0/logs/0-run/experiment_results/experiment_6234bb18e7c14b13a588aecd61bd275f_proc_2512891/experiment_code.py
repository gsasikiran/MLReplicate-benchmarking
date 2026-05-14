import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Synthetic dataset generation
class ReviewDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = np.random.rand(num_samples, 10).astype(np.float32)  # Features
        self.labels = (self.data.sum(axis=1) > 5).astype(
            np.float32
        )  # Simple binary classification as RQI

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.data[idx]),
            "labels": torch.tensor(self.labels[idx]).unsqueeze(0),
        }


# Define model
class ReviewerQualityModel(nn.Module):
    def __init__(self):
        super(ReviewerQualityModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Data preparation
dataset = ReviewDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, optimizer
model = ReviewerQualityModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

experiment_data = {
    "peer_review_experiment": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    experiment_data["peer_review_experiment"]["losses"]["train"].append(avg_loss)

    print(f"Epoch {epoch + 1}: training_loss = {avg_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
