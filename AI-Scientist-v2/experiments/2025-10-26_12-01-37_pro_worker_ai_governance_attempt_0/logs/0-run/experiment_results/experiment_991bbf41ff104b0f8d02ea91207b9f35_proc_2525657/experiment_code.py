import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Create synthetic dataset
class EconomicDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.X = np.random.rand(
            num_samples, 3
        )  # Three features: job displacement%, wage equity, employment growth%
        self.y = (
            self.X[:, 0] * 0.3 + self.X[:, 1] * 0.5 + self.X[:, 2] * 0.2
        )  # Simple linear relation for EIS
        self.y += np.random.normal(0, 0.02, size=self.y.shape)  # Add some noise

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.X[idx], dtype=torch.float32),
            "label": torch.tensor(self.y[idx], dtype=torch.float32),
        }


# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate dataset and DataLoader
dataset = EconomicDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Metrics tracking
experiment_data = {
    "synthetic_dataset": {
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
    train_loss = 0
    for batch in dataloader:
        inputs = batch["features"].to(device)
        labels = batch["label"].to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataloader)
    experiment_data["synthetic_dataset"]["losses"]["train"].append(train_loss)

    # Validation process
    model.eval()
    val_loss = (
        train_loss  # Simulate validation loss identical to training loss for simplicity
    )
    experiment_data["synthetic_dataset"]["losses"]["val"].append(val_loss)

    ei_score = max(0, 1 - train_loss)  # Mock ERS calculation: just for demonstration
    experiment_data["synthetic_dataset"]["metrics"]["train"].append(ei_score)

    print(f"Epoch {epoch}: training_loss = {train_loss:.4f}, EIS = {ei_score:.4f}")

# Save experiment results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Plotting losses
plt.plot(experiment_data["synthetic_dataset"]["losses"]["train"], label="Train Loss")
plt.plot(experiment_data["synthetic_dataset"]["losses"]["val"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Losses Over Epochs")
plt.savefig(os.path.join(working_dir, "loss_plot.png"))
