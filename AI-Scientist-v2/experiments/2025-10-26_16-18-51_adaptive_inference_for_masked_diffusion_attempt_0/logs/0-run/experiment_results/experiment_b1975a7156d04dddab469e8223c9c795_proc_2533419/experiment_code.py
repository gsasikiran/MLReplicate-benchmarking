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

optimizers = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "Adagrad": optim.Adagrad,
}

experiment_data = {
    "use_of_different_optimizers": {
        "sudoku": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

num_epochs = 10

for optimizer_name, optimizer in optimizers.items():
    print(f"Training with optimizer: {optimizer_name}")
    model = SudokuModel().to(device)
    criterion = nn.MSELoss()
    optimizer_instance = optimizer(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            optimizer_instance.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_instance.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_data["use_of_different_optimizers"]["sudoku"]["losses"][
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
        experiment_data["use_of_different_optimizers"]["sudoku"]["losses"][
            "val"
        ].append(val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
