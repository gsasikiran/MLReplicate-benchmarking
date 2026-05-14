import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Generate synthetic data
np.random.seed(0)
num_samples = 1000
job_displacement = np.random.uniform(0, 1, num_samples)
compensation_fairness = np.random.uniform(0, 1, num_samples)
equitable_access = np.random.uniform(0, 1, num_samples)
impact_scores = (
    (1 - job_displacement) * 0.4 + compensation_fairness * 0.3 + equitable_access * 0.3
)


# Define the dataset class
class ProWorkerDataset(Dataset):
    def __init__(self, features, targets):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"features": self.x[idx], "target": self.y[idx]}


# Prepare the dataset and dataloaders
dataset = ProWorkerDataset(
    np.column_stack((job_displacement, compensation_fairness, equitable_access)),
    impact_scores,
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
experiment_data = {
    "pro_worker": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        inputs = batch["features"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    experiment_data["pro_worker"]["losses"]["train"].append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["features"].to(device)
            targets = batch["target"].to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs.view(-1), targets).item()

    val_loss /= len(val_loader)
    experiment_data["pro_worker"]["losses"]["val"].append(val_loss)

    # Calculate Pro-Worker Governance Impact Score (higher is better, based on validation loss)
    impact_score = (
        1 - val_loss
    )  # Assuming lower loss indicates better governance impact
    experiment_data["pro_worker"]["metrics"]["val"].append(impact_score)

    print(
        f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, validation_loss = {val_loss:.4f}, Pro-Worker Governance Impact Score = {impact_score:.4f}"
    )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
