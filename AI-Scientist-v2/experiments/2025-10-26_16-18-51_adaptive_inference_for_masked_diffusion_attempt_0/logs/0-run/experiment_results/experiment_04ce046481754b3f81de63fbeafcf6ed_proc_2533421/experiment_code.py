import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SudokuDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = np.random.randint(1, 10, (num_samples, 81))
        self.labels = self.data.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SudokuModel(nn.Module):
    def __init__(self):
        super(SudokuModel, self).__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 81)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

dataset = SudokuDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

experiment_data = {
    "regularization": {
        "with_l2": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "without_l2": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

num_epochs = 15

# Training without L2 regularization
model = SudokuModel().to(device)
optimizer_no_l2 = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Training without L2 regularization")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)

        optimizer_no_l2.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_no_l2.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    experiment_data["regularization"]["without_l2"]["losses"]["train"].append(
        train_loss
    )

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    experiment_data["regularization"]["without_l2"]["losses"]["val"].append(val_loss)
    print(
        f"Epoch {epoch + 1}/{num_epochs}: validation_loss (without L2) = {val_loss:.4f}"
    )

# Training with L2 regularization
model = SudokuModel().to(device)
optimizer_with_l2 = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

print("Training with L2 regularization")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)

        optimizer_with_l2.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_with_l2.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    experiment_data["regularization"]["with_l2"]["losses"]["train"].append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    experiment_data["regularization"]["with_l2"]["losses"]["val"].append(val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss (with L2) = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
