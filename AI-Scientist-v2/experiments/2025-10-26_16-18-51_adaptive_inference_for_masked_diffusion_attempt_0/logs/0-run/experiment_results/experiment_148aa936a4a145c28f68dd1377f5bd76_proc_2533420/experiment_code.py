import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SudokuDataset(Dataset):
    def __init__(self, num_samples=1000, complexity="easy"):
        self.complexity = complexity
        self.data = self.generate_sudoku_samples(num_samples)

    def generate_sudoku_samples(self, num_samples):
        if self.complexity == "easy":
            return np.random.randint(1, 4, (num_samples, 81))  # Easier values
        elif self.complexity == "medium":
            return np.random.randint(
                1, 7, (num_samples, 81)
            )  # Medium complexity values
        elif self.complexity == "hard":
            return np.random.randint(
                1, 10, (num_samples, 81)
            )  # Harder complexity values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.data[idx], dtype=torch.float32),
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

datasets = {
    "easy": SudokuDataset(1000, "easy"),
    "medium": SudokuDataset(1000, "medium"),
    "hard": SudokuDataset(1000, "hard"),
}

experiment_data = {"multiple_synthetic_datasets": {}}

for dataset_name, dataset in datasets.items():
    print(f"Training on {dataset_name} dataset...")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SudokuModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data["multiple_synthetic_datasets"][dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
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
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "train"
            ].append(train_loss)

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
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "val"
            ].append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: validation_loss = {val_loss:.4f}")

    # Collect predictions and ground truth (optional, depending on needs)
    experiment_data["multiple_synthetic_datasets"][dataset_name]["predictions"].append(
        outputs.cpu().numpy()
    )
    experiment_data["multiple_synthetic_datasets"][dataset_name]["ground_truth"].append(
        labels.cpu().numpy()
    )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
