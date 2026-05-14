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


# Baseline Model definition
class SudokuModel(nn.Module):
    def __init__(self):
        super(SudokuModel, self).__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 81)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Ablation Model definition (with additional layers)
class ExtendedSudokuModel(nn.Module):
    def __init__(self):
        super(ExtendedSudokuModel, self).__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 81)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Prepare dataset and loaders
dataset = SudokuDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Experiment data dictionary
experiment_data = {
    "baseline_model": {
        "sudoku": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
    "additional_layers_model": {
        "sudoku": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}


# Function to train and validate a model
def train_and_validate(model, num_epochs, label):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

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
        experiment_data[label]["sudoku"]["losses"]["train"].append(train_loss)

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
        experiment_data[label]["sudoku"]["losses"]["val"].append(val_loss)
        print(
            f"{label} Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}"
        )


# Train and validate baseline model
print("Training Baseline Model")
baseline_model = SudokuModel().to(device)
train_and_validate(baseline_model, num_epochs=10, label="baseline_model")

# Train and validate extended model
print("Training Additional Layers Model")
extended_model = ExtendedSudokuModel().to(device)
train_and_validate(extended_model, num_epochs=10, label="additional_layers_model")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
