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
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Generate synthetic dataset
class PeerReviewDataset(Dataset):
    def __init__(self, size, with_interactions=False):
        self.data = np.random.rand(size, 2)  # Two independent features
        if with_interactions:
            interaction_terms = self.data[:, 0] * self.data[:, 1]
            self.data = np.hstack(
                (self.data, interaction_terms.reshape(-1, 1))
            )  # Add interaction term
        self.labels = np.random.rand(size)  # RQI labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx], dtype=torch.float32).to(device),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32).to(device),
        }


# Create synthetic dataset
dataset = PeerReviewDataset(size=1000, with_interactions=False)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Hyperparameter tuning for learning rates
learning_rates = [0.001, 0.01, 0.1]
batch_size = 64  # Static batch size for simplicity
experiment_data = {
    "feature_interaction": {
        "no_interaction": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "with_interaction": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

epochs = 10

# Training without interaction terms
for lr in learning_rates:
    print(f"Training without interaction terms using learning rate: {lr}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleNN(input_size=2).to(device)  # Two independent features
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
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
        experiment_data["feature_interaction"]["no_interaction"]["losses"][
            "train"
        ].append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch["features"], batch["label"]
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        experiment_data["feature_interaction"]["no_interaction"]["losses"][
            "val"
        ].append(avg_val_loss)

        rqi = 1 - avg_val_loss
        experiment_data["feature_interaction"]["no_interaction"]["metrics"][
            "train"
        ].append(rqi)

        print(
            f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}"
        )

    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch["features"], batch["label"]
            outputs = model(features).squeeze()
            experiment_data["feature_interaction"]["no_interaction"][
                "predictions"
            ].extend(outputs.cpu().numpy())
            experiment_data["feature_interaction"]["no_interaction"][
                "ground_truth"
            ].extend(labels.cpu().numpy())

# Training with interaction terms
dataset_interaction = PeerReviewDataset(size=1000, with_interactions=True)
train_dataset_interaction, val_dataset_interaction = random_split(
    dataset_interaction, [train_size, val_size]
)

for lr in learning_rates:
    print(f"Training with interaction terms using learning rate: {lr}")
    train_loader = DataLoader(
        train_dataset_interaction, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset_interaction, batch_size=batch_size, shuffle=False
    )

    model = SimpleNN(input_size=3).to(device)  # Three features including interaction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
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
        experiment_data["feature_interaction"]["with_interaction"]["losses"][
            "train"
        ].append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, labels = batch["features"], batch["label"]
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        experiment_data["feature_interaction"]["with_interaction"]["losses"][
            "val"
        ].append(avg_val_loss)

        rqi = 1 - avg_val_loss
        experiment_data["feature_interaction"]["with_interaction"]["metrics"][
            "train"
        ].append(rqi)

        print(
            f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}"
        )

    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch["features"], batch["label"]
            outputs = model(features).squeeze()
            experiment_data["feature_interaction"]["with_interaction"][
                "predictions"
            ].extend(outputs.cpu().numpy())
            experiment_data["feature_interaction"]["with_interaction"][
                "ground_truth"
            ].extend(labels.cpu().numpy())

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
