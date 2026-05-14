import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Define a simple synthetic Sudoku dataset
class SudokuDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = np.random.randint(1, 10, (num_samples, 81))  # 81 cells for Sudoku
        self.labels = self.data.copy()  # Simulating label as the same for simplicity

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# Normalize function for min-max scaling
def min_max_scaling(data):
    return (data - data.min(axis=1, keepdims=True)) / (
        data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True)
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Model definition
class SudokuModel(nn.Module):
    def __init__(self):
        super(SudokuModel, self).__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 81)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Prepare dataset and loaders
dataset = SudokuDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = SudokuModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data dictionary
experiment_data = {
    "input_normalization": {
        "sudoku": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
    "no_normalization": {
        "sudoku": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

num_epochs_list = [5, 10, 15, 20]

# Training with input normalization
for num_epochs in num_epochs_list:
    print(f"Training with input normalization and num_epochs: {num_epochs}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            inputs = min_max_scaling(inputs.cpu().numpy()).astype(
                np.float32
            )  # Normalize inputs
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data["input_normalization"]["sudoku"]["losses"]["train"].append(
            train_loss
        )

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)
                inputs = min_max_scaling(inputs.cpu().numpy()).astype(np.float32)
                inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        experiment_data["input_normalization"]["sudoku"]["losses"]["val"].append(
            val_loss
        )
        print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}")

# Reset model and optimizer for non-normalized training
model = SudokuModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training without input normalization
for num_epochs in num_epochs_list:
    print(f"Training without input normalization and num_epochs: {num_epochs}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data["no_normalization"]["sudoku"]["losses"]["train"].append(
            train_loss
        )

        # Validation
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
        experiment_data["no_normalization"]["sudoku"]["losses"]["val"].append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
