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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Model definition with customizable activation function
class SudokuModel(nn.Module):
    def __init__(self, activation_fn):
        super(SudokuModel, self).__init__()
        self.fc1 = nn.Linear(81, 128)
        self.fc2 = nn.Linear(128, 81)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        return self.fc2(x)


# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Prepare dataset and loaders
dataset = SudokuDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Experiment data dictionary
experiment_data = {
    "different_activation_functions": {
        "sudoku": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Epoch configurations for tuning
num_epochs_list = [5, 10, 15, 20]

# Activation functions to explore
activation_functions = {
    "ReLU": torch.relu,
    "Sigmoid": torch.sigmoid,
    "Tanh": torch.tanh,
}

for activation_name, activation_fn in activation_functions.items():
    print(f"Training with activation function: {activation_name}")

    # Initialize model, loss function, and optimizer
    model = SudokuModel(activation_fn).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            experiment_data["different_activation_functions"]["sudoku"]["losses"][
                "train"
            ].append(train_loss)

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
            experiment_data["different_activation_functions"]["sudoku"]["losses"][
                "val"
            ].append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
