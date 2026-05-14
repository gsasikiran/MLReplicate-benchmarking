import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic dataset creation
class ReviewDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = np.random.rand(num_samples, 5)  # 5 features
        self.labels = (
            self.data.sum(axis=1) + np.random.rand(num_samples)
        ) / 6  # RQI score

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx], dtype=torch.float).to(device),
            "label": torch.tensor(self.labels[idx], dtype=torch.float).to(device),
        }


# Model definition with variable hidden layers
class RQIModel(nn.Module):
    def __init__(self, hidden_units, num_layers):
        super(RQIModel, self).__init__()
        layers = []
        input_size = 5
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            input_size = hidden_units
        layers.append(nn.Linear(hidden_units, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Hyperparameters for tuning
hidden_units = 10
num_layers = 2  # This will represent the additional hidden layer

# Prepare dataset
dataset = ReviewDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = RQIModel(hidden_units, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data storage
experiment_data = {
    "hyperparam_tuning_additional_hidden_layer": {
        "RQI_experiment": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Training loop
for epoch in range(100):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        features = batch["features"]
        labels = batch["label"]
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    experiment_data["hyperparam_tuning_additional_hidden_layer"]["RQI_experiment"][
        "losses"
    ]["train"].append(train_loss)
    print(f"Epoch {epoch}: training_loss = {train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            features = batch["features"]
            labels = batch["label"]
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    experiment_data["hyperparam_tuning_additional_hidden_layer"]["RQI_experiment"][
        "losses"
    ]["val"].append(val_loss)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
