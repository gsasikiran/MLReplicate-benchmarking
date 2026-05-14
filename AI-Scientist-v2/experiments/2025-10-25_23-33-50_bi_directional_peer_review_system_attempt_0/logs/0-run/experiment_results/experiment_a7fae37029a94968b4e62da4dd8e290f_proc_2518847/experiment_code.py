import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

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


# Generate synthetic dataset for different distributions
class SyntheticDataset(Dataset):
    def __init__(self, size, distribution):
        if distribution == "uniform":
            self.data = np.random.rand(size, 2)  # Uniform distribution within [0, 1)
        elif distribution == "normal":
            self.data = np.random.normal(
                0, 1, (size, 2)
            )  # Normal distribution with mean 0, std 1
        elif distribution == "bimodal":
            self.data = np.vstack(
                (
                    np.random.normal(-2, 0.5, (size // 2, 2)),
                    np.random.normal(2, 0.5, (size // 2, 2)),
                )
            )  # Bimodal distribution
        self.labels = np.random.rand(size)  # RQI labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx], dtype=torch.float32).to(device),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32).to(device),
        }


# Hyperparameter tuning and model evaluation
learning_rates = [0.001, 0.01, 0.1]
batch_size = 64
epochs = 10

experiment_data = {
    "ablation_study": {
        "uniform": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "normal": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "bimodal": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

for distribution in ["uniform", "normal", "bimodal"]:
    print(f"Evaluating distribution: {distribution}")
    dataset = SyntheticDataset(size=1000, distribution=distribution)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features, labels = batch["features"], batch["label"]
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            experiment_data["ablation_study"][distribution]["losses"]["train"].append(
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
            experiment_data["ablation_study"][distribution]["losses"]["val"].append(
                avg_val_loss
            )

            rqi = 1 - avg_val_loss
            experiment_data["ablation_study"][distribution]["metrics"]["train"].append(
                rqi
            )

            print(
                f"Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, RQI = {rqi:.4f}"
            )

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                features, labels = batch["features"], batch["label"]
                outputs = model(features).squeeze()
                experiment_data["ablation_study"][distribution]["predictions"].extend(
                    outputs.cpu().numpy()
                )
                experiment_data["ablation_study"][distribution]["ground_truth"].extend(
                    labels.cpu().numpy()
                )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
