import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset

# Setting up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


# Create synthetic dataset
dataset = PeerReviewDataset(size=1000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Hyperparameter tuning for learning rates
learning_rates = [0.001, 0.01, 0.1]
batch_size = 64
experiment_data = {
    "both_features": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "author_rating_only": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
    "review_score_only": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

epochs = 10


def train_model(input_dim, lr, current_ablation):
    model = SimpleNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            features, labels = batch["features"], batch["label"]
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        experiment_data[current_ablation]["losses"]["train"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features, labels = batch["features"], batch["label"]
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        experiment_data[current_ablation]["losses"]["val"].append(avg_val_loss)

        rqi = 1 - avg_val_loss
        experiment_data[current_ablation]["metrics"]["train"].append(rqi)
        print(
            f"Ablation {current_ablation} - Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}"
        )

    # Collect predictions and ground truths
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            features, labels = batch["features"], batch["label"]
            outputs = model(features).squeeze()
            experiment_data[current_ablation]["predictions"].extend(
                outputs.cpu().numpy()
            )
            experiment_data[current_ablation]["ground_truth"].extend(
                labels.cpu().numpy()
            )


# Train with both features
current_ablation = "both_features"
for lr in learning_rates:
    print(f"Training with both features and learning rate: {lr}")
    train_model(input_dim=2, lr=lr, current_ablation=current_ablation)

# Train with author rating only
current_ablation = "author_rating_only"
for lr in learning_rates:
    print(f"Training with author rating only and learning rate: {lr}")
    dataset.data = np.random.rand(1000, 1)  # Use only one feature
    train_model(input_dim=1, lr=lr, current_ablation=current_ablation)

# Train with review score only
current_ablation = "review_score_only"
for lr in learning_rates:
    print(f"Training with review score only and learning rate: {lr}")
    dataset.data = np.random.rand(1000, 1)  # Use only one feature
    train_model(input_dim=1, lr=lr, current_ablation=current_ablation)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
