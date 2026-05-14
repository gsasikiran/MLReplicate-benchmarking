import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Define a simple synthetic Sudoku dataset
class SudokuDataset(Dataset):
    def __init__(self, num_samples=1000, randomness_level=1):
        self.data = np.random.randint(1, 10, (num_samples, 81))  # 81 cells for Sudoku
        # Simulate more structured patterns based on randomness level
        if randomness_level == 2:
            self.data[0 : num_samples // 2, :] = np.random.choice(
                [1, 2, 3, 4, 5, 6, 7, 8, 9], (num_samples // 2, 81)
            )
        elif randomness_level == 3:
            self.data[0 : num_samples // 3, :] = np.random.choice(
                [1, 2, 3, 4, 5, 6, 7, 8, 9], (num_samples // 3, 81)
            )
            self.data[num_samples // 3 : (num_samples * 2) // 3, :] = np.random.randint(
                1, 10, (num_samples // 3, 81)
            )  # Random
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

# Initialize experiment data structure
experiment_data = {
    "multi_dataset_evaluation": {
        "dataset_randomness_1": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_randomness_2": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_randomness_3": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Training settings
num_epochs_list = [5, 10]
batch_size = 32

# Evaluate across three different datasets with varying randomness levels
for randomness_level in range(1, 4):
    dataset = SudokuDataset(randomness_level=randomness_level)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SudokuModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for num_epochs in num_epochs_list:
        print(
            f"Training with num_epochs: {num_epochs} on dataset_randomness_{randomness_level}"
        )

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
            experiment_data["multi_dataset_evaluation"][
                f"dataset_randomness_{randomness_level}"
            ]["losses"]["train"].append(train_loss)

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
            experiment_data["multi_dataset_evaluation"][
                f"dataset_randomness_{randomness_level}"
            ]["losses"]["val"].append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
