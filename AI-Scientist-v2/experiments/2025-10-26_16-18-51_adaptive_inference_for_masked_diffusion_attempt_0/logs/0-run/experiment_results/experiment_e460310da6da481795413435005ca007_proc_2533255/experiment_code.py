import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SudokuDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = np.random.randint(1, 10, (num_samples, 81))
        self.labels = self.data.copy()  # Simulated label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class KakuroDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Simulating Kakuro dataset for illustration purposes
        self.data = np.random.randint(1, 10, (num_samples, 64))
        self.labels = self.data.copy()  # Simulated label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class NonogramsDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Simulating Nonograms dataset for illustration purposes
        self.data = np.random.randint(1, 2, (num_samples, 100))  # Binary grid
        self.labels = self.data.copy()  # Simulated label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LogicPuzzlesModel(nn.Module):
    def __init__(self):
        super(LogicPuzzlesModel, self).__init__()
        self.fc1 = nn.Linear(81, 128)  # For Sudoku
        self.fc2 = nn.Linear(128, 81)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Prepare datasets and loaders
sudoku_dataset = SudokuDataset()
kakuro_dataset = KakuroDataset()
nonograms_dataset = NonogramsDataset()

train_loader = DataLoader(sudoku_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(sudoku_dataset, batch_size=32, shuffle=False)

model = LogicPuzzlesModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

experiment_data = {
    "Sudoku": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

num_epochs_list = [5, 10, 15, 20]

for num_epochs in num_epochs_list:
    print(f"Training with num_epochs: {num_epochs}")

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
        experiment_data["Sudoku"]["losses"]["train"].append(train_loss)

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
        experiment_data["Sudoku"]["losses"]["val"].append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
