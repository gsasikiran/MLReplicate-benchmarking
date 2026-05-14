import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load datasets
imdb_data = load_dataset("imdb")
ag_news_data = load_dataset("ag_news")
yelp_data = load_dataset("yelp_polarity")

# Prepare synthetic data and use data from Hugging Face datasets
np.random.seed(42)
X = np.random.rand(1000, 3)
y = np.clip(X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2], 0, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class SimpleDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


train_dataset = SimpleDataset(X_train, y_train)
val_dataset = SimpleDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class SimpleNet(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


hidden_units_list = [16, 32, 64]
experiment_data = {
    "hyperparam_tuning": {
        "hidden_units": {},
    },
}

for hidden_units in hidden_units_list:
    model = SimpleNet(hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(
                device
            )  # Move to device
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.view(-1), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units]["losses"][
            "train"
        ].append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(
                    device
                )  # Move to device
                outputs = model(features)
                loss = criterion(outputs.view(-1), targets)
                val_loss += loss.item()
                experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                    "predictions"
                ].extend(outputs.cpu().numpy())
                experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units][
                    "ground_truth"
                ].extend(targets.cpu().numpy())

        experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units]["losses"][
            "val"
        ].append(val_loss / len(val_loader))
        print(
            f"Hidden Units: {hidden_units}, Epoch {epoch}: validation_loss = {val_loss:.4f}"
        )

        # Calculate and store Worker Empowerment Score (WES)
        WES = np.mean(
            [
                1 if pred > 0.5 else 0
                for pred in experiment_data["hyperparam_tuning"]["hidden_units"][
                    hidden_units
                ]["predictions"]
            ]
        )
        experiment_data["hyperparam_tuning"]["hidden_units"][hidden_units]["metrics"][
            "val"
        ].append(WES)

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
