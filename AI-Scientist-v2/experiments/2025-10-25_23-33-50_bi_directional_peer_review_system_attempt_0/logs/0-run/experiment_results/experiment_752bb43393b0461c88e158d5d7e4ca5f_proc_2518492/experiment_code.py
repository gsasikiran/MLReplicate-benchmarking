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


class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PeerReviewDataset(Dataset):
    def __init__(self, size):
        self.data = np.random.rand(size, 2)  # Two features
        self.labels = np.random.rand(size)  # RQI labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx], dtype=torch.float32).to(device),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32).to(device),
        }


# Load three datasets from Hugging Face
dataset_1 = load_dataset("glue", "mrpc", split="train")
dataset_2 = load_dataset("glue", "cola", split="train")
dataset_3 = load_dataset("glue", "sst2", split="train")  # Adding a third dataset

# Create synthetic dataset
dataset = PeerReviewDataset(size=1000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Hyperparameter tuning
learning_rates = [0.001, 0.01, 0.1]
batch_size = 32  # Reduced batch size for complex model
experiment_data = {
    "peer_review": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "RIS": [],
    }
}

epochs = 10

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ComplexNN().to(device)
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
        experiment_data["peer_review"]["losses"]["train"].append(avg_train_loss)

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

        # Review Quality Index (RQI) and Reviewer Impact Score (RIS) calculation
        rqi = 1 - avg_val_loss  # Assume RQI inversely relates to loss
        ris = rqi * 0.5  # Simple correlation, to be refined
        experiment_data["peer_review"]["metrics"]["train"].append(rqi)
        experiment_data["peer_review"]["RIS"].append(ris)

        print(
            f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}, RIS = {ris:.4f}"
        )

    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch["features"], batch["label"]
            outputs = model(features).squeeze()
            experiment_data["peer_review"]["predictions"].extend(outputs.cpu().numpy())
            experiment_data["peer_review"]["ground_truth"].extend(labels.cpu().numpy())

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
