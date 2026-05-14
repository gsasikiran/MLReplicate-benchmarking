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
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Generate synthetic dataset
class PeerReviewDataset(Dataset):
    def __init__(self, size):
        self.data = np.random.rand(size, 2)
        self.labels = np.random.rand(size)

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
    "gradient_clipping": {
        "without_clipping": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "with_clipping": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

epochs = 10


# Function to train the model
def train_model(
    model, train_loader, val_loader, optimizer, criterion, epoch, clip_grad=False
):
    for batch in train_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        features, labels = batch["features"], batch["label"]
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return loss.item()


for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Without gradient clipping
    model = SimpleNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        avg_train_loss = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epoch,
            clip_grad=False,
        )
        experiment_data["gradient_clipping"]["without_clipping"]["losses"][
            "train"
        ].append(avg_train_loss)

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
        experiment_data["gradient_clipping"]["without_clipping"]["losses"][
            "val"
        ].append(avg_val_loss)

    # Collect predictions
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            features, labels = batch["features"], batch["label"]
            outputs = model(features).squeeze()
            experiment_data["gradient_clipping"]["without_clipping"][
                "predictions"
            ].extend(outputs.cpu().numpy())
            experiment_data["gradient_clipping"]["without_clipping"][
                "ground_truth"
            ].extend(labels.cpu().numpy())

    # With gradient clipping
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        avg_train_loss = train_model(
            model, train_loader, val_loader, optimizer, criterion, epoch, clip_grad=True
        )
        experiment_data["gradient_clipping"]["with_clipping"]["losses"]["train"].append(
            avg_train_loss
        )

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
        experiment_data["gradient_clipping"]["with_clipping"]["losses"]["val"].append(
            avg_val_loss
        )

    # Collect predictions
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            features, labels = batch["features"], batch["label"]
            outputs = model(features).squeeze()
            experiment_data["gradient_clipping"]["with_clipping"]["predictions"].extend(
                outputs.cpu().numpy()
            )
            experiment_data["gradient_clipping"]["with_clipping"][
                "ground_truth"
            ].extend(labels.cpu().numpy())

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
