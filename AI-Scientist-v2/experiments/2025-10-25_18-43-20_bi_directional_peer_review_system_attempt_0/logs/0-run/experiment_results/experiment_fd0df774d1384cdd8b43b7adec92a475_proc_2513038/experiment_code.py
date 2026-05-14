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


# Model definition
class RQIModel(nn.Module):
    def __init__(self, hidden_layer_size):
        super(RQIModel, self).__init__()
        self.fc1 = nn.Linear(5, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Prepare dataset
dataset = ReviewDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Hyperparameter tuning for hidden_layer_size
hidden_layer_sizes = [5, 10, 15, 20]  # Testing different sizes
experiment_data = {"hyperparam_tuning_hidden_layer_size": {}}

for hidden_layer_size in hidden_layer_sizes:
    # Initialize model, loss function, and optimizer
    model = RQIModel(hidden_layer_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data["hyperparam_tuning_hidden_layer_size"][
        f"size_{hidden_layer_size}"
    ] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
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
        experiment_data["hyperparam_tuning_hidden_layer_size"][
            f"size_{hidden_layer_size}"
        ]["losses"]["train"].append(train_loss)
        print(
            f"hidden_layer_size {hidden_layer_size}, Epoch {epoch}: training_loss = {train_loss:.4f}"
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features = batch["features"]
                labels = batch["label"]
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        experiment_data["hyperparam_tuning_hidden_layer_size"][
            f"size_{hidden_layer_size}"
        ]["losses"]["val"].append(val_loss)
        print(
            f"hidden_layer_size {hidden_layer_size}, Epoch {epoch}: validation_loss = {val_loss:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
