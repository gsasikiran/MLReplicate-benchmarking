import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic dataset class
class WorkerImpactDataset(Dataset):
    def __init__(self, num_samples=1000):
        np.random.seed(42)
        self.X = np.random.rand(
            num_samples, 3
        )  # 3 features: displacement, wage_change, satisfaction
        self.y = (
            self.X[:, 0] * -0.5
            + self.X[:, 1] * 0.3
            + self.X[:, 2] * 0.2
            + np.random.normal(0, 0.1, num_samples)
        )  # WIS
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Initialize dataset and dataloaders
dataset = WorkerImpactDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Simple neural network model
class ImpactModel(nn.Module):
    def __init__(self):
        super(ImpactModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize model, optimizer, and loss function
model = ImpactModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Data structure for logging
experiment_data = {
    "synthetic_data": {
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
    total_loss_train = 0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()

    avg_loss_train = total_loss_train / len(train_loader)
    experiment_data["synthetic_data"]["losses"]["train"].append(avg_loss_train)

    # Validation
    model.eval()
    total_loss_val = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss_val += loss.item()

    avg_loss_val = total_loss_val / len(val_loader)
    experiment_data["synthetic_data"]["losses"]["val"].append(avg_loss_val)
    print(f"Epoch {epoch + 1}: validation_loss = {avg_loss_val:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Visualization of predictions vs ground truth
model.eval()
with torch.no_grad():
    all_predictions = []
    all_ground_truth = []
    for batch in val_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_predictions.extend(outputs.cpu().numpy())
        all_ground_truth.extend(targets.cpu().numpy())

all_predictions = np.array(all_predictions)
all_ground_truth = np.array(all_ground_truth)

plt.figure(figsize=(10, 5))
plt.scatter(all_ground_truth, all_predictions, alpha=0.5)
plt.xlabel("Ground Truth WIS")
plt.ylabel("Predicted WIS")
plt.title("Ground Truth vs Predicted Worker Impact Score (WIS)")
plt.savefig(os.path.join(working_dir, "wis_predictions.png"))
plt.show()
